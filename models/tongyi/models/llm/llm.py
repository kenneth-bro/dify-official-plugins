import base64
import logging
import json
import os
import tempfile
import uuid
from collections.abc import Generator
from http import HTTPStatus
from pathlib import Path
from typing import Optional, Union, cast

import requests
from dashscope import Generation, MultiModalConversation, get_tokenizer
from dashscope.api_entities.dashscope_response import GenerationResponse
from dashscope.common.error import (
    AuthenticationError,
    InvalidParameter,
    RequestFailure,
    ServiceUnavailableError,
    UnsupportedHTTPMethod,
    UnsupportedModel,
)
from dify_plugin.entities.model import (
    AIModelEntity,
    FetchFrom,
    I18nObject,
    ModelFeature,
    ModelPropertyKey,
    ModelType,
    ParameterRule,
    ParameterType,
)
from dify_plugin.entities.model.llm import (
    LLMMode,
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
)
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    DocumentPromptMessageContent,
    ImagePromptMessageContent,
    PromptMessage,
    PromptMessageContentType,
    PromptMessageRole,
    PromptMessageTool,
    SystemPromptMessage,
    TextPromptMessageContent,
    ToolPromptMessage,
    UserPromptMessage,
    VideoPromptMessageContent,
    AudioPromptMessageContent,
)
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from dify_plugin.config.logger_format import plugin_logger_handler
from dify_plugin.interfaces.model.large_language_model import LargeLanguageModel
from openai import OpenAI
from models._common import get_http_base_address
from ..constant import BURY_POINT_HEADER

logger = logging.getLogger(__name__)
# [CUSTOM-i] Route plugin logs through Dify's stdout JSON log event handler so plugin-daemon can capture them
logger.setLevel(logging.INFO)
if plugin_logger_handler not in logger.handlers:
    logger.addHandler(plugin_logger_handler)
logger.propagate = False

MODEL_ALIAS_ENV_NAME = "TONGYI_MODEL_ALIAS_MAP"  # [CUSTOM-i] Deployment-level model alias env var.
DEFAULT_MODEL_ALIAS_MAP = ""  # [CUSTOM-i] Empty by default; aliases only apply when explicitly configured.
PROMPT_LOG_PREVIEW_CHARS = 300  # [CUSTOM-i] Keep prompt previews short enough for daemon logs.
REQUEST_LOG_TEXT_PREVIEW_CHARS = 500  # [CUSTOM-i] Truncate final DashScope request snapshots in debug logs.
# [CUSTOM-i] DashScope explicit context cache customization. Manual mode is system-prompt only.
CONTEXT_CACHE_MODE_ENV_NAME = "TONGYI_CONTEXT_CACHE_MODE"  # [CUSTOM-i] off|manual.
CONTEXT_CACHE_OFF = "off"
CONTEXT_CACHE_MANUAL = "manual"
CONTEXT_CACHE_START_TAG = "<cache>"
CONTEXT_CACHE_END_TAG = "</cache>"
CONTEXT_CACHE_MAX_MARKERS = 4
CONTEXT_CACHE_SUPPORTED_MODELS = {
    "qwen3.6-plus",
    "qwen3.5-plus",
    "qwen3.5-flash",
    "qwen-plus",
}


class TongyiLargeLanguageModel(LargeLanguageModel):
    tokenizers = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._temp_files = []

    @staticmethod
    def _parse_model_alias_map(alias_map_config: Optional[str]) -> dict[str, str]:
        # [CUSTOM-i] Parse model alias pairs: qwen3.5-plus=qwen3.5-plus-2026-04-20,qwen-plus=qwen-plus-latest.
        if not alias_map_config:
            return {}

        alias_map = {}
        for item in alias_map_config.split(","):
            if "=" not in item:
                continue
            source, target = item.split("=", 1)
            source = source.strip()
            target = target.strip()
            if source and target:
                alias_map[source] = target
        return alias_map

    @classmethod
    def _resolve_actual_model(cls, requested_model: str) -> str:
        # [CUSTOM-i] Allow deployment-level model aliasing without changing Dify workflow model selections.
        alias_map = cls._parse_model_alias_map(os.getenv(MODEL_ALIAS_ENV_NAME, DEFAULT_MODEL_ALIAS_MAP))
        return alias_map.get(requested_model, requested_model)

    @staticmethod
    def _prompt_text_preview(prompt_messages: list[PromptMessage]) -> str:
        # [CUSTOM-i] Log only textual prompt fragments; skip binary/media payloads and keep the line compact.
        fragments: list[str] = []
        for prompt_message in prompt_messages:
            role = getattr(prompt_message.role, "value", str(prompt_message.role))
            content = prompt_message.content
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text_parts = [
                    data
                    for item in content
                    if getattr(item, "type", None) == PromptMessageContentType.TEXT
                    and isinstance(data := getattr(item, "data", None), str)
                ]
                text = "\n".join(text_parts)
            else:
                text = str(content)

            text = " ".join(text.split())
            if text:
                fragments.append(f"{role}: {text}")

        return "\n".join(fragments)[:PROMPT_LOG_PREVIEW_CHARS]

    @classmethod
    def _sanitize_dashscope_log_value(cls, value):
        # [CUSTOM-i] Mask secrets and truncate long strings before emitting DashScope debug logs.
        if isinstance(value, dict):
            sanitized = {}
            for key, item in value.items():
                if str(key).lower() in {"api_key", "dashscope_api_key", "authorization"}:
                    sanitized[key] = cls._mask_secret_for_log(item)
                else:
                    sanitized[key] = cls._sanitize_dashscope_log_value(item)
            return sanitized
        if isinstance(value, list):
            return [cls._sanitize_dashscope_log_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._sanitize_dashscope_log_value(item) for item in value)
        if isinstance(value, str):
            if len(value) > REQUEST_LOG_TEXT_PREVIEW_CHARS:
                return f"{value[:REQUEST_LOG_TEXT_PREVIEW_CHARS]}...<truncated>"
            return value
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if hasattr(value, "model_dump"):
            return cls._sanitize_dashscope_log_value(value.model_dump())
        if hasattr(value, "to_dict"):
            return cls._sanitize_dashscope_log_value(value.to_dict())
        if hasattr(value, "__dict__"):
            return cls._sanitize_dashscope_log_value(
                {
                    key: item
                    for key, item in vars(value).items()
                    if not key.startswith("_")
                }
            )
        return repr(value)

    @staticmethod
    def _mask_secret_for_log(value) -> str:
        # [CUSTOM-i] Show enough of DashScope API keys for credential routing checks without exposing the full secret.
        if value is None:
            return ""
        text = str(value)
        if text.lower().startswith("bearer "):
            return f"Bearer {TongyiLargeLanguageModel._mask_secret_for_log(text[7:])}"
        if len(text) <= 10:
            return "******"
        return f"{text[:5]}******{text[-5:]}"

    @classmethod
    def _dashscope_request_for_log(
        cls,
        params: dict,
        headers: dict,
        stream: bool,
        incremental_output: bool,
        base_address: str,
        user: Optional[str],
    ) -> dict:
        # [CUSTOM-i] Capture the final provider request after message conversion and cache_control injection.
        request_params = dict(params)
        request_params["headers"] = headers
        request_params["stream"] = stream
        request_params["incremental_output"] = incremental_output
        request_params["base_address"] = base_address
        request_params["user"] = user
        return cls._sanitize_dashscope_log_value(request_params)

    @classmethod
    def _dashscope_usage_for_log(cls, response: GenerationResponse):
        # [CUSTOM-i] Preserve raw DashScope usage fields, including cache-hit counters when the SDK returns them.
        return cls._sanitize_dashscope_log_value(getattr(response, "usage", None))

    @staticmethod
    def _normalize_context_cache_mode(mode: Optional[str]) -> str:
        # [CUSTOM-i] Keep unknown cache modes fail-closed so old workflows continue without cache markers.
        normalized = (mode or CONTEXT_CACHE_OFF).strip().lower()
        if normalized in {
            CONTEXT_CACHE_OFF,
            CONTEXT_CACHE_MANUAL,
        }:
            return normalized
        return CONTEXT_CACHE_OFF

    @staticmethod
    def _supports_dashscope_context_cache(model: str) -> bool:
        # [CUSTOM-i] Runtime gate mirrors the enabled predefined model YAMLs instead of all qwen aliases.
        return model in CONTEXT_CACHE_SUPPORTED_MODELS

    @staticmethod
    def _text_cache_block(text: str, cached: bool = False) -> dict:
        # [CUSTOM-i] DashScope explicit cache marker uses Anthropic-style cache_control blocks.
        block = {"type": "text", "text": text}
        if cached:
            block["cache_control"] = {"type": "ephemeral"}
        return block

    @classmethod
    def _split_manual_cache_blocks(cls, text: str) -> Optional[list[dict]]:
        # [CUSTOM-i] Strip <cache> tags before provider calls and mark only their enclosed text.
        if CONTEXT_CACHE_START_TAG not in text or CONTEXT_CACHE_END_TAG not in text:
            return None

        blocks = []
        rest = text
        while CONTEXT_CACHE_START_TAG in rest:
            before, after_start = rest.split(CONTEXT_CACHE_START_TAG, 1)
            if CONTEXT_CACHE_END_TAG not in after_start:
                if before:
                    blocks.append(cls._text_cache_block(before))
                blocks.append(cls._text_cache_block(CONTEXT_CACHE_START_TAG + after_start))
                return blocks

            cached, rest = after_start.split(CONTEXT_CACHE_END_TAG, 1)
            if before:
                blocks.append(cls._text_cache_block(before))
            if cached:
                blocks.append(cls._text_cache_block(cached, cached=True))

        if rest:
            blocks.append(cls._text_cache_block(rest))
        return blocks

    @staticmethod
    def _has_manual_cache_tags(messages: list[dict]) -> bool:
        # [CUSTOM-i] Manual cache tags are recognized only in system prompts.
        for message in messages:
            if message.get("role") != "system":
                continue
            content = message.get("content")
            if isinstance(content, str):
                if CONTEXT_CACHE_START_TAG in content and CONTEXT_CACHE_END_TAG in content:
                    return True
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    text = block.get("text")
                    if (
                        isinstance(text, str)
                        and CONTEXT_CACHE_START_TAG in text
                        and CONTEXT_CACHE_END_TAG in text
                    ):
                        return True
        return False

    @classmethod
    def _limit_context_cache_markers(cls, messages: list[dict]) -> None:
        # [CUSTOM-i] DashScope accepts up to four cache markers; keep the newest system markers.
        marker_blocks = []
        for message in messages:
            if message.get("role") != "system":
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    marker_blocks.append(block)

        for block in marker_blocks[:-CONTEXT_CACHE_MAX_MARKERS]:
            block.pop("cache_control", None)

    @classmethod
    def _apply_manual_cache_tags(cls, messages: list[dict]) -> None:
        # [CUSTOM-i] If any system <cache> tag exists, only tagged system text is cached.
        for message in messages:
            if message.get("role") != "system":
                continue
            content = message.get("content")
            if isinstance(content, str):
                blocks = cls._split_manual_cache_blocks(content)
                if blocks:
                    message["content"] = blocks
            elif isinstance(content, list):
                converted_content = []
                changed = False
                for block in content:
                    if not isinstance(block, dict) or not isinstance(block.get("text"), str):
                        converted_content.append(block)
                        continue

                    blocks = cls._split_manual_cache_blocks(block["text"])
                    if blocks:
                        converted_content.extend(blocks)
                        changed = True
                    else:
                        converted_content.append(block)
                if changed:
                    message["content"] = converted_content

    @classmethod
    def _apply_whole_text_context_cache(cls, messages: list[dict]) -> None:
        # [CUSTOM-i] Manual mode fallback: no system <cache> tags means cache the whole system prompt.
        for message in messages:
            if message.get("role") != "system":
                continue
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = [cls._text_cache_block(content, cached=True)]
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict) or not isinstance(block.get("text"), str):
                        continue
                    block["type"] = "text"
                    block["cache_control"] = {"type": "ephemeral"}

    @classmethod
    def _apply_manual_context_cache(cls, messages: list[dict]) -> None:
        # [CUSTOM-i] Implements the user-facing manual rule for system prompts only.
        if cls._has_manual_cache_tags(messages):
            cls._apply_manual_cache_tags(messages)
        else:
            cls._apply_whole_text_context_cache(messages)

    @classmethod
    def _apply_dashscope_context_cache(
        cls, messages: list[dict], mode: str, model: str
    ) -> list[dict]:
        # [CUSTOM-i] Apply DashScope cache_control after Dify message conversion and before SDK invocation.
        if mode == CONTEXT_CACHE_OFF or not cls._supports_dashscope_context_cache(model):
            return messages

        if mode == CONTEXT_CACHE_MANUAL:
            cls._apply_manual_context_cache(messages)

        cls._limit_context_cache_markers(messages)
        return messages

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        """
        Invoke large language model

        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param model_parameters: model parameters
        :param tools: tools for tool calling
        :param stop: stop words
        :param stream: is stream response
        :param user: unique user id
        :return: full response or stream response chunk generator result
        """
        return self._generate(
            model,
            credentials,
            prompt_messages,
            model_parameters,
            tools,
            stop,
            stream,
            user,
        )

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        """
        Get number of tokens for given prompt messages

        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param tools: tools for tool calling
        :return:
        """
        if self.get_customizable_model_schema(model, credentials) is not None:
            return 0
        if model in {"qwen-turbo-chat", "qwen-plus-chat"}:
            model = model.replace("-chat", "")
        if model == "farui-plus":
            model = "qwen-farui-plus"
        if model in self.tokenizers:
            tokenizer = self.tokenizers[model]
        else:
            tokenizer = get_tokenizer(model)
            self.tokenizers[model] = tokenizer
        tokens = tokenizer.encode(self._convert_messages_to_prompt(prompt_messages))
        return len(tokens)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials

        :param model: model name
        :param credentials: model credentials
        :return:
        """
        try:
            self._generate(
                model=model,
                credentials=credentials,
                prompt_messages=[UserPromptMessage(content="ping")],
                model_parameters={"temperature": 0.5},
                stream=False,
            )
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    def _generate(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        """
        Invoke large language model

        :param model: model name
        :param credentials: credentials
        :param prompt_messages: prompt messages
        :param tools: tools for tool calling
        :param model_parameters: model parameters
        :param stop: stop words
        :param stream: is stream response
        :param user: unique user id
        :return: full response or stream response chunk generator result
        """
        # [CUSTOM-i] Private Dify context moved into credentials because the plugin SDK filters model_parameters.
        workflow_run_id = credentials.pop("dify_workflow_run_id", None) or credentials.pop("_dify_workflow_run_id", None)
        credentials_kwargs = self._to_credential_kwargs(credentials)
        mode = self.get_model_mode(model, credentials)
        if model in {"qwen-turbo-chat", "qwen-plus-chat"}:
            model = model.replace("-chat", "")

        extra_model_kwargs = {}
        if tools:
            extra_model_kwargs["tools"] = self._convert_tools(tools)
        if stop:
            extra_model_kwargs["stop"] = stop

        response_format = model_parameters.get("response_format")
        if response_format:
            model_parameters["response_format"] = {"type": response_format}

        if model.startswith("qwen-mt"):
            source_lang = model_parameters.pop("source_lang", None)
            target_lang = model_parameters.pop("target_lang", None)
            domains = model_parameters.pop("domains", None)
            model_parameters["translation_options"] = {
                "source_lang": source_lang,
                "target_lang": target_lang,
                "domains": domains,
            }
            # The Qwen-MT model does not support incremental streaming output at this time.
            stream = False
            if len(prompt_messages) > 1:
                prompt_messages = prompt_messages[-1:]
            if prompt_messages[-1].role != PromptMessageRole.USER:
                raise ValueError(
                    "There is one and only one User Message in the messages array."
                )
        # For models that support enable_thinking parameter, explicitly set it to False if not provided
        # This overrides API-level defaults where some models default to thinking mode enabled
        # Reference: https://help.aliyun.com/zh/model-studio/deep-thinking
        thinking_capable_models = {
            # Qwen Plus/Turbo series (default: thinking disabled, but explicit False ensures consistency)
            "qwen-plus-latest", "qwen-plus-2025-04-28",
            "qwen-turbo-latest", "qwen-turbo-2025-04-28",
            "qwen-flash", "qwen-flash-2025-07-28",
            # Qwen3 Max series (default: thinking disabled)
            "qwen3-max-2026-01-23", "qwen3-max-preview",
            # Qwen3.5/3.6 series (default: thinking ENABLED - must explicitly disable)
            "qwen3.6-plus", "qwen3.6-plus-2026-04-02",
            "qwen3.6-flash", "qwen3.6-flash-2026-04-16",
            "qwen3.5-plus", "qwen3.5-plus-2026-02-15",
            "qwen3.5-flash", "qwen3.5-flash-2026-02-23",
            # GLM series (default: thinking ENABLED - must explicitly disable)
            "glm-5.1", "glm-5", "glm-4.7", "glm-4.6", "glm-4.5", "glm-4.5-air",
            # DeepSeek series (default: thinking disabled)
            "deepseek-v3.2", "deepseek-v3.2-exp", "deepseek-v3.1",
        }
        if model in thinking_capable_models and "enable_thinking" not in model_parameters:
            model_parameters["enable_thinking"] = False

        extra_headers_str = ''
        if model_parameters.get('extra_headers',''):
            extra_headers_str = model_parameters.pop('extra_headers')
        # [CUSTOM-i] Provider-specific cache switch is consumed locally and never forwarded to DashScope.
        context_cache_mode = self._normalize_context_cache_mode(
            model_parameters.pop(
                "context_cache_mode",
                os.getenv(CONTEXT_CACHE_MODE_ENV_NAME, CONTEXT_CACHE_OFF),
            )
        )
        # [CUSTOM-i] Backward-compatible fallback; never pass this private Dify parameter to DashScope.
        workflow_run_id = workflow_run_id or model_parameters.pop("_dify_workflow_run_id", None)

        model_schema = self.get_model_schema(model, credentials)
        selected_model = model
        # [CUSTOM-i] Replace only the DashScope request model after resolving Dify schema/features from the selected model.
        model = self._resolve_actual_model(model)
        params = {
            "model": model,
            **model_parameters,
            **credentials_kwargs,
            **extra_model_kwargs,
        }

        incremental_output = False if tools else stream

        thinking_business_qwen3 = model in (
            "qwen-plus-latest",
            "qwen-plus-2025-04-28",
            "qwen-turbo-latest",
            "qwen-turbo-2025-04-28",
            "qwen-flash", "qwen-flash-2025-07-28",
            "qwen3-max-2026-01-23", "qwen3-max-preview",
            "qwen3.6-plus", "qwen3.6-plus-2026-04-02",
            "qwen3.5-plus", "qwen3.5-plus-2026-02-15",
            "qwen3.5-flash", "qwen3.5-flash-2026-02-23",
        ) and model_parameters.get("enable_thinking", False)

        # GLM models with thinking capability (default: thinking enabled)
        thinking_glm = model in ("glm-5", "glm-4.7", "glm-4.6", "glm-4.5", "glm-4.5-air") \
                        and model_parameters.get("enable_thinking", False)

        # Kimi models with thinking capability: kimi-k2.5 (when enable_thinking=true) and kimi-k2-thinking
        thinking_kimi = (
            model == "kimi-k2.5" and model_parameters.get("enable_thinking", False)
        ) or model == "kimi-k2-thinking"

        # Qwen3 business edition (Thinking Mode), Qwen3 open-source edition (excluding coder, max, and 3.5 variants), QwQ, QVQ, Kimi, and GLM thinking models only supports streaming output.
        # Note: qwen3-coder-xx, qwen3-max-xx, and qwen3.5-xx models support non-streaming output.
        # Note: qwen3-coder-xx, qwen3-max-xx, and qwen3.5-xx models support non-streaming output.
        qwen3_requires_stream = model.startswith("qwen3-") and not model.startswith(
            ("qwen3-coder", "qwen3-max", "qwen3.5-")
        )
        common_force_condition = (
            thinking_business_qwen3 or qwen3_requires_stream or thinking_kimi or thinking_glm
        )
        if common_force_condition or model.startswith(("qwq-", "qvq-")):
            stream = True
        # Qwen3 business edition (Thinking Mode), Qwen3 open-source edition (excluding coder and max variants), QwQ, QVQ, Kimi, and GLM thinking models only supports incremental_output set to True.
        if common_force_condition or model.startswith(("qwq-", "qvq-")):
            incremental_output = True

        base_address = get_http_base_address(credentials)

        # The parameter `enable_omni_output_audio_url` must be set to true when using the Omni model in non-streaming mode.
        if model.startswith("qwen3-omni-") and not stream:
            params["enable_omni_output_audio_url"] = True

        # [CUSTOM-i] Log model request parameters except prompt messages and credentials for troubleshooting.
        debug_logging = str(credentials.get("debug_logging", "false")).lower() == "true"
        if debug_logging:
            request_params = {
                key: value
                for key, value in params.items()
                if key not in {"messages"}
            }
            request_params["user"] = user
            request_params["stream"] = stream
            request_params["incremental_output"] = incremental_output
            request_params["base_address"] = base_address
            logger.warning(
                "🔍 [dashscope-params] workflow_run_id=%s model=%s prompt_preview=%r params=%r",
                workflow_run_id,
                model,
                self._prompt_text_preview(prompt_messages),
                self._sanitize_dashscope_log_value(request_params),
            )

        if ModelFeature.VISION in (model_schema.features or []):
            params["messages"] = self._convert_prompt_messages_to_tongyi_messages(
                credentials, prompt_messages, rich_content=True
            )
            # [CUSTOM-i] Disable Alibaba Cloud content inspection (绿网) and merge with market bury point header
            _call_headers = self._get_market_bury_point_header(params["messages"], extra_headers_str)
            _call_headers["X-DashScope-DataInspection"] = '{"input":"disable","output":"disable"}'
            # [CUSTOM-i] Preserve selected Dify model as the cache support gate even when request model is aliased.
            params["messages"] = self._apply_dashscope_context_cache(
                params["messages"], context_cache_mode, selected_model
            )
            if debug_logging:
                logger.warning(
                    "🔍 [dashscope-request] workflow_run_id=%s model=%s request=%r",
                    workflow_run_id,
                    model,
                    self._dashscope_request_for_log(
                        params, _call_headers, stream, incremental_output, base_address, user
                    ),
                )
            response = MultiModalConversation.call(
                **params,
                stream=stream,
                headers=_call_headers,
                incremental_output=incremental_output,
                base_address=base_address,
            )
        else:
            params["messages"] = self._convert_prompt_messages_to_tongyi_messages(
                credentials, prompt_messages
            )
            # [CUSTOM-i] Disable Alibaba Cloud content inspection (绿网) and merge with market bury point header
            _call_headers = self._get_market_bury_point_header(params["messages"], extra_headers_str)
            _call_headers["X-DashScope-DataInspection"] = '{"input":"disable","output":"disable"}'
            # [CUSTOM-i] Preserve selected Dify model as the cache support gate even when request model is aliased.
            params["messages"] = self._apply_dashscope_context_cache(
                params["messages"], context_cache_mode, selected_model
            )
            if debug_logging:
                logger.warning(
                    "🔍 [dashscope-request] workflow_run_id=%s model=%s request=%r",
                    workflow_run_id,
                    model,
                    self._dashscope_request_for_log(
                        params, _call_headers, stream, incremental_output, base_address, user
                    ),
                )
            response = Generation.call(
                **params,
                headers=_call_headers,
                result_format="message",
                stream=stream,
                incremental_output=incremental_output,
                base_address=base_address,
            )
        if stream:
            return self._handle_generate_stream_response(
                model,
                credentials,
                response,
                prompt_messages,
                incremental_output,
                debug_logging=debug_logging,
                workflow_run_id=workflow_run_id,
            )
        return self._handle_generate_response(
            model,
            credentials,
            response,
            prompt_messages,
            debug_logging=debug_logging,
            workflow_run_id=workflow_run_id,
        )

    def _handle_generate_response(
        self,
        model: str,
        credentials: dict,
        response: GenerationResponse,
        prompt_messages: list[PromptMessage],
        debug_logging: bool = False,
        workflow_run_id: Optional[str] = None,
    ) -> LLMResult:
        """
        Handle llm response

        :param model: model name
        :param credentials: credentials
        :param response: response
        :param prompt_messages: prompt messages
        :return: llm response
        """
        try:
            if response.status_code not in {200, HTTPStatus.OK}:
                # Get request_id (if present) and forward it to the error handler.
                request_id = getattr(response, "request_id", None)
                self._handle_error_response(
                    response.status_code, response.message, model, request_id
                )

            resp_content = response.output.choices[0].message.content
            # special for qwen-vl
            if isinstance(resp_content, list):
                resp_content = resp_content[0]["text"]
            assistant_prompt_message = AssistantPromptMessage(
                content=resp_content,
                tool_calls=response.output.choices[0].message.get("tool_calls", []),
            )
            usage = self._calc_response_usage(
                model,
                credentials,
                response.usage.input_tokens,
                response.usage.output_tokens,
            )
            result = LLMResult(
                model=model,
                message=assistant_prompt_message,
                prompt_messages=prompt_messages,
                usage=usage,
            )
            # [CUSTOM-i] Log request_id for tracing with Alibaba Cloud support (controlled by debug_logging credential)
            if debug_logging:
                logger.warning(
                    "🔍 [dashscope] workflow_run_id=%s model=%s request_id=%s usage=%r full_text=%r",
                    workflow_run_id,
                    model,
                    response.request_id,
                    self._dashscope_usage_for_log(response),
                    str(resp_content),
                )
            return result
        finally:
            self._cleanup_temp_files()

    def _handle_tool_call_stream(self, response, tool_calls, incremental_output):
        tool_calls_stream = response.output.choices[0].message["tool_calls"]
        for tool_call_stream in tool_calls_stream:
            idx = tool_call_stream.get("index")
            if idx >= len(tool_calls):
                tool_calls.append(tool_call_stream)
            else:
                if tool_call_stream.get("function"):
                    func_name = tool_call_stream.get("function").get("name")
                    tool_call_obj = tool_calls[idx]
                    if func_name:
                        if incremental_output:
                            tool_call_obj["function"]["name"] += func_name
                        else:
                            tool_call_obj["function"]["name"] = func_name
                    args = tool_call_stream.get("function").get("arguments")
                    if args:
                        if incremental_output:
                            tool_call_obj["function"]["arguments"] += args
                        else:
                            tool_call_obj["function"]["arguments"] = args

    def _handle_generate_stream_response(
        self,
        model: str,
        credentials: dict,
        responses: Generator[GenerationResponse, None, None],
        prompt_messages: list[PromptMessage],
        incremental_output: bool,
        debug_logging: bool = False,
        workflow_run_id: Optional[str] = None,
    ) -> Generator:
        """
        Handle llm stream response

        :param model: model name
        :param credentials: credentials
        :param responses: response
        :param prompt_messages: prompt messages
        :param incremental_output: is incremental output
        :return: llm response chunk generator result
        """
        is_reasoning = False
        # This is used to handle unincremental output correctly
        full_text = ""
        tool_calls = []
        try:
            for index, response in enumerate(responses):
                if response.status_code not in {200, HTTPStatus.OK}:
                    # Get request_id (if present) and forward it to the error handler.
                    request_id = getattr(response, "request_id", None)
                    self._handle_error_response(
                        response.status_code, response.message, model, request_id
                    )

                resp_finish_reason = response.output.choices[0].finish_reason
                if resp_finish_reason is not None and resp_finish_reason != "null":
                    resp_content = response.output.choices[0].message.content
                    assistant_prompt_message = AssistantPromptMessage(content="")
                    if "tool_calls" in response.output.choices[0].message:
                        self._handle_tool_call_stream(
                            response, tool_calls, incremental_output
                        )
                    elif resp_content:
                        if isinstance(resp_content, list):
                            resp_content = resp_content[0]["text"]
                        if incremental_output:
                            assistant_prompt_message.content = resp_content
                            full_text += resp_content
                        else:
                            assistant_prompt_message.content = resp_content.replace(
                                full_text, "", 1
                            )
                            full_text = resp_content
                    elif is_reasoning:
                        assistant_prompt_message.content = "\n</think>"
                        full_text += "\n</think>"
                    if tool_calls:
                        message_tool_calls = []
                        for tool_call_obj in tool_calls:
                            message_tool_call = AssistantPromptMessage.ToolCall(
                                id=tool_call_obj["function"]["name"],
                                type="function",
                                function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                                    name=tool_call_obj["function"]["name"],
                                    arguments=tool_call_obj["function"]["arguments"],
                                ),
                            )
                            message_tool_calls.append(message_tool_call)
                        assistant_prompt_message.tool_calls = message_tool_calls
                    usage = response.usage
                    usage = self._calc_response_usage(
                        model, credentials, usage.input_tokens, usage.output_tokens
                    )
                    # [CUSTOM-i] Log request_id and merged response content for tracing with Alibaba Cloud support
                    if debug_logging:
                        logger.warning(
                            "🔍 [dashscope] workflow_run_id=%s model=%s request_id=%s usage=%r full_text=%r",
                            workflow_run_id,
                            model,
                            response.request_id,
                            self._dashscope_usage_for_log(response),
                            full_text,
                        )
                    yield LLMResultChunk(
                        model=model,
                        prompt_messages=prompt_messages,
                        delta=LLMResultChunkDelta(
                            index=index,
                            message=assistant_prompt_message,
                            finish_reason=resp_finish_reason,
                            usage=usage,
                        ),
                    )
                else:
                    message = response.output.choices[0].message

                    resp_content, is_reasoning = (
                        self._wrap_thinking_by_reasoning_content(message, is_reasoning)
                    )

                    content_to_yield = []
                    if resp_content:
                        if incremental_output:
                            delta = resp_content
                            full_text += delta
                        else:
                            delta = resp_content.replace(full_text, "", 1)
                            full_text = resp_content
                        content_to_yield.append(delta)

                    if "tool_calls" in message:
                        if is_reasoning:
                            content_to_yield.append("\n</think>")
                            # In incremental mode (stream=True), full_text accumulates the generated content.
                            # In non-incremental mode, full_text tracks the raw API response state for delta calculation.
                            # Since "\n</think>" is synthesized locally and not part of the API response,
                            # we must NOT update full_text in non-incremental mode to avoid sync issues.
                            if incremental_output:
                                full_text += "\n</think>"
                            is_reasoning = False
                        self._handle_tool_call_stream(
                            response, tool_calls, incremental_output
                        )

                    if content_to_yield:
                        assistant_prompt_message = AssistantPromptMessage(
                            content="".join(content_to_yield)
                        )
                        yield LLMResultChunk(
                            model=model,
                            prompt_messages=prompt_messages,
                            delta=LLMResultChunkDelta(
                                index=index, message=assistant_prompt_message
                            ),
                        )
        finally:
            self._cleanup_temp_files()

    def _to_credential_kwargs(self, credentials: dict) -> dict:
        """
        Transform credentials to kwargs for model instance

        :param credentials:
        :return:
        """
        credentials_kwargs = {"api_key": credentials["dashscope_api_key"]}
        return credentials_kwargs

    def _convert_one_message_to_text(self, message: PromptMessage) -> str:
        """
        Convert a single message to a string.

        :param message: PromptMessage to convert.
        :return: String representation of the message.
        """
        human_prompt = "\n\nHuman:"
        ai_prompt = "\n\nAssistant:"
        content = message.content
        if isinstance(message, UserPromptMessage):
            if isinstance(content, str):
                message_text = f"{human_prompt} {content}"
            elif isinstance(content, list):
                message_text = ""
                for sub_message in content:
                    if sub_message.type == PromptMessageContentType.TEXT:
                        message_text = f"{human_prompt} {sub_message.data}"
                        break
            else:
                raise TypeError(
                    f"[convert_one_message_to_text] Unexpected content type: {type(content)}"
                )
        elif isinstance(message, AssistantPromptMessage):
            message_text = f"{ai_prompt} {content}"
        elif isinstance(message, SystemPromptMessage | ToolPromptMessage):
            message_text = content
        else:
            raise ValueError(f"Got unknown type {message}")
        return message_text

    def _convert_messages_to_prompt(self, messages: list[PromptMessage]) -> str:
        """
        Format a list of messages into a full prompt for the Anthropic model

        :param messages: List of PromptMessage to combine.
        :return: Combined string with necessary human_prompt and ai_prompt tags.
        """
        messages = messages.copy()
        text = "".join(
            (self._convert_one_message_to_text(message) for message in messages)
        )
        return text.rstrip()

    def _convert_prompt_messages_to_tongyi_messages(
        self,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        rich_content: bool = False,
    ) -> list[dict]:
        """
        Convert prompt messages to tongyi messages

        :param prompt_messages: prompt messages
        :return: tongyi messages
        """
        tongyi_messages = []
        for prompt_message in prompt_messages:
            if isinstance(prompt_message, SystemPromptMessage):
                tongyi_messages.append(
                    {
                        "role": "system",
                        "content": (
                            prompt_message.content
                            if not rich_content
                            else [{"text": prompt_message.content}]
                        ),
                    }
                )
            elif isinstance(prompt_message, UserPromptMessage):
                if isinstance(prompt_message.content, str):
                    tongyi_messages.append(
                        {
                            "role": "user",
                            "content": (
                                prompt_message.content
                                if not rich_content
                                else [{"text": prompt_message.content}]
                            ),
                        }
                    )
                else:
                    user_messages = []
                    file_id_list = []
                    for message_content in prompt_message.content:
                        if message_content.type == PromptMessageContentType.TEXT:
                            message_content = cast(
                                TextPromptMessageContent, message_content
                            )
                            sub_message_dict = {"text": message_content.data}
                            user_messages.append(sub_message_dict)
                        elif message_content.type == PromptMessageContentType.IMAGE:
                            message_content = cast(
                                ImagePromptMessageContent, message_content
                            )
                            image_url = message_content.data
                            if message_content.data.startswith("data:"):
                                image_url = self._save_base64_to_file(
                                    message_content.data
                                )
                            sub_message_dict = {"image": image_url}
                            user_messages.append(sub_message_dict)
                        elif message_content.type == PromptMessageContentType.VIDEO:
                            message_content = cast(
                                VideoPromptMessageContent, message_content
                            )
                            video_url = message_content.data
                            if message_content.data.startswith("data:"):
                                video_url = self._save_base64_to_file(
                                    message_content.data
                                )
                            sub_message_dict = {"video": video_url}
                            user_messages.append(sub_message_dict)
                        elif message_content.type == PromptMessageContentType.AUDIO:
                            message_content = cast(
                                AudioPromptMessageContent, message_content
                            )
                            audio_data = message_content.data
                            if not audio_data:
                                raise ValueError("Audio content cannot be empty.")
                            if audio_data.startswith("data:"):
                                audio_data = self._save_base64_to_file(audio_data)
                            sub_message_dict = {"audio": audio_data}
                            user_messages.append(sub_message_dict)
                        elif message_content.type == PromptMessageContentType.DOCUMENT:
                            message_content = cast(
                                DocumentPromptMessageContent, message_content
                            )
                            file_id = self._upload_file_to_tongyi(
                                credentials, message_content
                            )
                            file_id_url = f"fileid://{file_id}"
                            file_id_list.append(file_id_url)
                    if len(file_id_list) > 0:
                        tongyi_messages.append(
                            {"role": "system", "content": ",".join(file_id_list)}
                        )
                    user_messages = sorted(user_messages, key=lambda x: "text" in x)
                    tongyi_messages.append({"role": "user", "content": user_messages})
            elif isinstance(prompt_message, AssistantPromptMessage):
                content = prompt_message.content
                if not content:
                    content = " "
                message = {
                    "role": "assistant",
                    "content": content if not rich_content else [{"text": content}],
                }
                if prompt_message.tool_calls:
                    message["tool_calls"] = [
                        tool_call.model_dump()
                        for tool_call in prompt_message.tool_calls
                    ]
                tongyi_messages.append(message)
            elif isinstance(prompt_message, ToolPromptMessage):
                tongyi_messages.append(
                    {
                        "role": "tool",
                        "content": prompt_message.content,
                        "name": prompt_message.tool_call_id,
                    }
                )
            else:
                raise ValueError(f"Got unknown type {prompt_message}")
        return tongyi_messages

    def _save_base64_to_file(self, base64_data: str) -> str:
        """
        Save base64 data to file
        'data:{upload_file.mime_type};base64,{encoded_string}'

        :param base64_data: base64 data
        :return: file path
        """
        (mime_type, encoded_string) = (
            base64_data.split(",")[0].split(";")[0].split(":")[1],
            base64_data.split(",")[1],
        )
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.{mime_type.split('/')[1]}")
        Path(file_path).write_bytes(base64.b64decode(encoded_string))
        self._temp_files.append(file_path)
        return f"file://{file_path}"

    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        for file_path in self._temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {file_path}: {e}")
        self._temp_files.clear()

    def _upload_file_to_tongyi(
        self, credentials: dict, message_content: DocumentPromptMessageContent
    ) -> str:
        """
        Upload file to Tongyi

        :param credentials: credentials for Tongyi
        :param message_content: message content to upload
        :return: file ID in Tongyi
        """
        client = OpenAI(
            api_key=credentials["dashscope_api_key"],
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        if credentials.get("use_international_endpoint", "false") == "true":
            client = OpenAI(
                api_key=credentials["dashscope_api_key"],
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            )
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file_path = temp_file.name
                if message_content.base64_data:
                    file_content = base64.b64decode(message_content.base64_data)
                    temp_file.write(file_content)
                else:
                    try:
                        response = requests.get(message_content.url, timeout=60)
                        response.raise_for_status()
                        temp_file.write(response.content)
                    except Exception as ex:
                        raise ValueError(
                            f"Failed to fetch data from url {message_content.url}, {ex}"
                        ) from ex
                temp_file.flush()
            # Close temp file first, then reopen with open() for OpenAI SDK compatibility
            with open(temp_file_path, "rb") as f:
                response = client.files.create(file=f, purpose="file-extract")
            return response.id
        finally:
            # Clean up temporary file after upload
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e:
                    logger.warning(
                        f"Failed to remove temporary file {temp_file_path}: {e}"
                    )

    def _convert_tools(self, tools: list[PromptMessageTool]) -> list[dict]:
        """
        Convert tools
        """
        tool_definitions = []
        for tool in tools:
            properties = tool.parameters["properties"]
            required_properties = tool.parameters["required"]
            properties_definitions = {}
            for p_key, p_val in properties.items():
                desc = p_val.get("description") or ""
                if "enum" in p_val:
                    desc += f"; Only accepts one of the following predefined options: [{', '.join(p_val['enum'])}]"
                properties_definitions[p_key] = {
                    "description": desc,
                    "type": p_val["type"],
                }
            tool_definition = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": properties_definitions,
                    "required": required_properties,
                },
            }
            tool_definitions.append(tool_definition)
        return tool_definitions

    def _wrap_thinking_by_reasoning_content(
        self, delta: dict, is_reasoning: bool
    ) -> tuple[str, bool]:
        """
        If the reasoning response is from delta.get("reasoning_content"), we wrap
        it with HTML think tag.
        :param delta: delta dictionary from LLM streaming response
        :param is_reasoning: is reasoning
        :return: tuple of (processed_content, is_reasoning)
        """

        content = delta.get("content") or ""
        if isinstance(content, list) and content:
            content = content[0].get("text") if isinstance(content[0], dict) else ""
        else:
            content = str(content)
        reasoning_content = delta.get("reasoning_content")
        try:
            if reasoning_content:
                try:
                    if isinstance(reasoning_content, list):
                        reasoning_content = "\n".join(map(str, reasoning_content))
                    elif not isinstance(reasoning_content, str):
                        reasoning_content = str(reasoning_content)

                    if not is_reasoning:
                        content = "<think>\n" + reasoning_content
                        is_reasoning = True
                    else:
                        content = reasoning_content
                except Exception as ex:
                    raise ValueError(
                        f"[wrap_thinking_by_reasoning_content-1] {ex}"
                    ) from ex
            elif is_reasoning and content:
                content = "\n</think>" + content
                is_reasoning = False
        except Exception as ex:
            raise ValueError(f"[wrap_thinking_by_reasoning_content-2] {ex}") from ex
        return content, is_reasoning

    def _handle_error_response(
        self, status_code: int, message: str, model: str = None, request_id: str = None
    ) -> None:
        """
        Handle error response based on HTTP status code

        :param status_code: HTTP status code
        :param message: error message
        :param model: model name (optional, for more detailed error messages)
        :param request_id: request id from Tongyi API response (optional)
        :raises: Appropriate InvokeError based on status code
        """
        if model:
            error_msg = f"Failed to invoke model {model}, status code: {status_code}, message: {message}"
        else:
            error_msg = message

        if request_id:
            error_msg += f", request_id: {request_id}"

        if status_code == 400:
            raise InvokeBadRequestError(error_msg)
        elif status_code == 401:
            raise InvokeAuthorizationError(error_msg)
        elif status_code == 403:
            raise InvokeAuthorizationError(error_msg)
        elif status_code == 422:
            raise InvokeBadRequestError(error_msg)
        elif status_code == 429:
            raise InvokeRateLimitError(error_msg)
        elif status_code >= 500:
            raise InvokeServerUnavailableError(error_msg)
        else:
            # For any other 4xx errors, treat as bad request
            if 400 <= status_code < 500:
                raise InvokeBadRequestError(error_msg)
            # For any other status codes, treat as server unavailable
            else:
                raise InvokeServerUnavailableError(error_msg)

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        """
        Map model invoke error to unified error
        The key is the error type thrown to the caller
        The value is the error type thrown by the model,
        which needs to be converted into a unified error type for the caller.

        :return: Invoke error mapping
        """
        return {
            InvokeConnectionError: [RequestFailure],
            InvokeServerUnavailableError: [ServiceUnavailableError],
            InvokeRateLimitError: [],
            InvokeAuthorizationError: [AuthenticationError],
            InvokeBadRequestError: [
                InvalidParameter,
                UnsupportedModel,
                UnsupportedHTTPMethod,
            ],
        }

    def get_customizable_model_schema(
        self, model: str, credentials: dict
    ) -> Optional[AIModelEntity]:
        """
        Architecture for defining customizable models

        :param model: model name
        :param credentials: model credentials
        :return: AIModelEntity or None
        """
        return AIModelEntity(
            model=model,
            label=I18nObject(en_US=model, zh_Hans=model),
            model_type=ModelType.LLM,
            features=(
                [
                    ModelFeature.TOOL_CALL,
                    ModelFeature.MULTI_TOOL_CALL,
                    ModelFeature.STREAM_TOOL_CALL,
                ]
                if credentials.get("function_calling_type") == "tool_call"
                else []
            ),
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                ModelPropertyKey.CONTEXT_SIZE: int(
                    credentials.get("context_size", 8000)
                ),
                ModelPropertyKey.MODE: LLMMode.CHAT.value,
            },
            parameter_rules=[
                ParameterRule(
                    name="temperature",
                    use_template="temperature",
                    label=I18nObject(en_US="Temperature", zh_Hans="温度"),
                    type=ParameterType.FLOAT,
                ),
                ParameterRule(
                    name="max_tokens",
                    use_template="max_tokens",
                    default=512,
                    min=1,
                    max=int(credentials.get("max_tokens", 1024)),
                    label=I18nObject(en_US="Max Tokens", zh_Hans="最大标记"),
                    type=ParameterType.INT,
                ),
                ParameterRule(
                    name="top_p",
                    use_template="top_p",
                    label=I18nObject(en_US="Top P", zh_Hans="Top P"),
                    type=ParameterType.FLOAT,
                ),
                ParameterRule(
                    name="top_k",
                    use_template="top_k",
                    label=I18nObject(en_US="Top K", zh_Hans="Top K"),
                    type=ParameterType.FLOAT,
                ),
                ParameterRule(
                    name="frequency_penalty",
                    use_template="frequency_penalty",
                    label=I18nObject(en_US="Frequency Penalty", zh_Hans="重复惩罚"),
                    type=ParameterType.FLOAT,
                ),
            ],
        )

    def _get_market_bury_point_header(self, messages: list[dict], extra_headers_str: str) -> dict:
        """
        Extract market bury point header information from messages

        This function parses system role messages in the messages list to extract productCode and buyerUid,
        constructs the bury point header, and cleans up the marketParams tag content from the original message.

        Args:
            messages (list[dict]): Message list, each element contains role and content fields

        Returns:
            dict: Bury point header information dictionary containing moduleCode, accountId and other fields;
                  If no valid information can be extracted, returns the default BURY_POINT_HEADER
        """
        res_bury_point_header = {}
        system_entries = [entry for entry in messages if entry["role"] == "system"]
        if system_entries:
            system_entry = system_entries[0].get("content", "")
            if system_entry:
                try:
                    system_entry_split = system_entry.split("||||||")
                    if len(system_entry_split) >= 2:
                        burn = system_entry_split[0].split(",")
                        bury_point_header = json.loads(
                            BURY_POINT_HEADER.get("x-dashscope-euid")
                        )
                        if len(burn) in (1, 2):
                            product_code = burn[0]
                            buyer_uid = burn[1] if len(burn) == 2 else ""
                            bury_point_header["moduleCode"] = product_code.strip()
                            bury_point_header["accountId"] = buyer_uid.strip()

                        system_entries[0]["content"] = "".join(system_entry_split[1:])
                        res_bury_point_header = {"x-dashscope-euid": json.dumps(bury_point_header)}
                except Exception:
                    res_bury_point_header = {}

        if extra_headers_str and res_bury_point_header == {}:
            try:
                # Replace non-breaking spaces and other special whitespace characters with regular spaces
                cleaned_str = extra_headers_str.replace('\xa0', ' ').replace('\u3000', ' ')
                res_bury_point_header = json.loads(cleaned_str)
            except Exception:
                res_bury_point_header = {}

        return res_bury_point_header
