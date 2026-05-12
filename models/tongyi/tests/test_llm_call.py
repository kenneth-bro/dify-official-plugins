import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from dify_plugin.config.integration_config import IntegrationConfig
from dify_plugin.core.entities.plugin.request import (
    ModelActions,
    ModelInvokeLLMRequest,
    PluginInvokeType,
)
from dify_plugin.entities.model import ModelType
from dify_plugin.entities.model.llm import LLMResultChunk
from dify_plugin.entities.model.message import SystemPromptMessage
from dify_plugin.integration.run import PluginRunner
import models.llm.llm as tongyi_llm
from models.llm.llm import (
    CONTEXT_CACHE_OFF,
    CONTEXT_CACHE_MANUAL,
    TongyiLargeLanguageModel,
)


EXCLUDED_MODELS = {
    "qwen2.5-1.5b-instruct",
    "qwen2.5-0.5b-instruct",
}


def get_all_models() -> list[str]:
    models_dir = Path(__file__).parent.parent / "models" / "llm"
    position_file = models_dir / "_position.yaml"
    if not position_file.exists():
        raise FileNotFoundError(f"Missing model position file: {position_file}")

    try:
        data = yaml.safe_load(position_file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {position_file}") from exc

    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(f"Expected a YAML list in {position_file}")

    models: list[str] = []
    for item in data:
        if isinstance(item, str) and item.strip():
            model_name = item.strip()
            if model_name in EXCLUDED_MODELS:
                continue
            models.append(model_name)
    return models


@pytest.mark.parametrize("model_name", get_all_models())
def test_llm_invoke(model_name: str) -> None:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY environment variable is required")

    plugin_path = os.getenv("PLUGIN_FILE_PATH")
    if not plugin_path:
        plugin_path = str(Path(__file__).parent.parent)

    payload = ModelInvokeLLMRequest(
        user_id="test_user",
        provider="tongyi",
        model_type=ModelType.LLM,
        model=model_name,
        credentials={"dashscope_api_key": api_key},
        prompt_messages=[{"role": "user", "content": "Say hello in one word."}],
        model_parameters={"max_tokens": 100},
        stop=None,
        tools=None,
        stream=True,
    )

    with PluginRunner(config=IntegrationConfig(), plugin_package_path=plugin_path) as runner:
        results: list[LLMResultChunk] = []
        for result in runner.invoke(
            access_type=PluginInvokeType.Model,
            access_action=ModelActions.InvokeLLM,
            payload=payload,
            response_type=LLMResultChunk,
        ):
            results.append(result)

        assert len(results) > 0, f"No results received for model {model_name}"

        full_content = "".join(
            r.delta.message.content for r in results if r.delta.message.content
        )
        assert len(full_content) > 0, f"Empty content for model {model_name}"


# [CUSTOM-i] Unit coverage for the Tongyi DashScope explicit context cache customization.
def test_invoke_applies_context_cache_to_generation_call(monkeypatch) -> None:
    captured = {}
    model = TongyiLargeLanguageModel.__new__(TongyiLargeLanguageModel)
    model._temp_files = []

    monkeypatch.setenv("TONGYI_MODEL_ALIAS_MAP", "qwen-plus=qwen-plus-0112")
    monkeypatch.setattr(model, "get_model_mode", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        model,
        "get_model_schema",
        lambda *_args, **_kwargs: SimpleNamespace(features=[]),
    )
    monkeypatch.setattr(
        model,
        "_handle_generate_response",
        lambda *_args, **_kwargs: "ok",
    )

    def fake_generation_call(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(tongyi_llm.Generation, "call", fake_generation_call)

    result = model._invoke(
        model="qwen-plus",
        credentials={"dashscope_api_key": "test-key"},
        prompt_messages=[SystemPromptMessage(content="stable system rules")],
        model_parameters={"context_cache_mode": CONTEXT_CACHE_MANUAL},
        stream=False,
    )

    assert result == "ok"
    assert captured["model"] == "qwen-plus-0112"
    assert captured["messages"] == [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "stable system rules",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]


def test_dashscope_request_log_includes_cache_control_and_redacts_secrets() -> None:
    request = TongyiLargeLanguageModel._dashscope_request_for_log(
        params={
            "model": "qwen-plus",
            "api_key": "secret-key",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "stable rules",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
        },
        headers={"Authorization": "Bearer secret", "X-DashScope-DataInspection": "{}"},
        stream=False,
        incremental_output=False,
        base_address="https://dashscope.aliyuncs.com/api/v1",
        user="test-user",
    )

    assert request["api_key"] == "***"
    assert request["headers"]["Authorization"] == "***"
    assert request["messages"][0]["content"][0]["cache_control"] == {
        "type": "ephemeral"
    }


def test_dashscope_usage_log_preserves_cache_usage_fields() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            prompt_tokens_details={"cached_tokens": 80},
        )
    )

    assert TongyiLargeLanguageModel._dashscope_usage_for_log(response) == {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "prompt_tokens_details": {"cached_tokens": 80},
    }


def test_context_cache_off_is_noop() -> None:
    messages = [
        {
            "role": "system",
            "content": "<cache>stable rules</cache>",
        }
    ]

    TongyiLargeLanguageModel._apply_dashscope_context_cache(
        messages, CONTEXT_CACHE_OFF, "qwen-plus"
    )

    assert messages == [
        {
            "role": "system",
            "content": "<cache>stable rules</cache>",
        }
    ]


def test_manual_context_cache_strips_cache_tags_and_marks_text_block() -> None:
    messages = [
        {
            "role": "system",
            "content": "before <cache>stable rules</cache> after",
        }
    ]

    TongyiLargeLanguageModel._apply_dashscope_context_cache(
        messages, CONTEXT_CACHE_MANUAL, "qwen-plus"
    )

    assert messages == [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "before "},
                {
                    "type": "text",
                    "text": "stable rules",
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": " after"},
            ],
        }
    ]


def test_manual_context_cache_marks_whole_text_when_no_cache_tags() -> None:
    messages = [
        {
            "role": "system",
            "content": "stable rules without explicit tags",
        },
        {
            "role": "user",
            "content": "runtime input",
        },
    ]

    TongyiLargeLanguageModel._apply_dashscope_context_cache(
        messages, CONTEXT_CACHE_MANUAL, "qwen-plus"
    )

    assert messages == [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "stable rules without explicit tags",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "user",
            "content": "runtime input",
        },
    ]


def test_manual_context_cache_does_not_mark_untagged_messages_when_tags_exist() -> None:
    messages = [
        {
            "role": "system",
            "content": "<cache>stable rules</cache>",
        },
        {
            "role": "user",
            "content": "runtime input",
        },
    ]

    TongyiLargeLanguageModel._apply_dashscope_context_cache(
        messages, CONTEXT_CACHE_MANUAL, "qwen-plus"
    )

    assert messages == [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "stable rules",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "user",
            "content": "runtime input",
        },
    ]


def test_manual_context_cache_only_applies_to_system_messages() -> None:
    messages = [
        {
            "role": "user",
            "content": "<cache>runtime input</cache>",
        },
        {
            "role": "assistant",
            "content": "assistant text",
        },
    ]

    TongyiLargeLanguageModel._apply_dashscope_context_cache(
        messages, CONTEXT_CACHE_MANUAL, "qwen-plus"
    )

    assert messages == [
        {
            "role": "user",
            "content": "<cache>runtime input</cache>",
        },
        {
            "role": "assistant",
            "content": "assistant text",
        },
    ]


def test_manual_context_cache_ignores_non_system_tags_when_system_has_no_tags() -> None:
    messages = [
        {
            "role": "system",
            "content": "stable system rules",
        },
        {
            "role": "user",
            "content": "<cache>runtime input</cache>",
        },
        {
            "role": "assistant",
            "content": "<cache>assistant text</cache>",
        },
        {
            "role": "tool",
            "content": "<cache>tool output</cache>",
        },
    ]

    TongyiLargeLanguageModel._apply_dashscope_context_cache(
        messages, CONTEXT_CACHE_MANUAL, "qwen-plus"
    )

    assert messages == [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "stable system rules",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "user",
            "content": "<cache>runtime input</cache>",
        },
        {
            "role": "assistant",
            "content": "<cache>assistant text</cache>",
        },
        {
            "role": "tool",
            "content": "<cache>tool output</cache>",
        },
    ]


def test_manual_context_cache_handles_system_rich_content() -> None:
    messages = [
        {
            "role": "system",
            "content": [
                {"text": "before <cache>stable rules</cache> after"},
                {"image": "https://example.com/image.png"},
            ],
        }
    ]

    TongyiLargeLanguageModel._apply_dashscope_context_cache(
        messages, CONTEXT_CACHE_MANUAL, "qwen-plus"
    )

    assert messages == [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "before "},
                {
                    "type": "text",
                    "text": "stable rules",
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": " after"},
                {"image": "https://example.com/image.png"},
            ],
        }
    ]


def test_manual_context_cache_skips_unsupported_non_qwen_models() -> None:
    messages = [
        {
            "role": "system",
            "content": "<cache>stable rules</cache>",
        }
    ]

    TongyiLargeLanguageModel._apply_dashscope_context_cache(
        messages, CONTEXT_CACHE_MANUAL, "glm-5"
    )

    assert messages[0]["content"] == "<cache>stable rules</cache>"


def test_manual_context_cache_skips_non_enabled_qwen_versions() -> None:
    messages = [
        {
            "role": "system",
            "content": "<cache>stable rules</cache>",
        }
    ]

    TongyiLargeLanguageModel._apply_dashscope_context_cache(
        messages, CONTEXT_CACHE_MANUAL, "qwen-plus-0112"
    )

    assert messages[0]["content"] == "<cache>stable rules</cache>"


def test_manual_context_cache_keeps_last_four_markers() -> None:
    messages = [
        {
            "role": "system",
            "content": "".join(f"<cache>block-{index}</cache>" for index in range(6)),
        }
    ]

    TongyiLargeLanguageModel._apply_dashscope_context_cache(
        messages, CONTEXT_CACHE_MANUAL, "qwen-plus"
    )

    cached_texts = [
        block["text"]
        for block in messages[0]["content"]
        if "cache_control" in block
    ]
    assert cached_texts == ["block-2", "block-3", "block-4", "block-5"]
    assert len(cached_texts) == 4
    assert [block["text"] for block in messages[0]["content"]] == [
        "block-0",
        "block-1",
        "block-2",
        "block-3",
        "block-4",
        "block-5",
    ]


def test_manual_context_cache_does_not_remove_non_system_existing_cache_control() -> None:
    messages = [
        {
            "role": "system",
            "content": "".join(f"<cache>block-{index}</cache>" for index in range(5)),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "runtime input",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ]

    TongyiLargeLanguageModel._apply_dashscope_context_cache(
        messages, CONTEXT_CACHE_MANUAL, "qwen-plus"
    )

    assert messages[1]["content"] == [
        {
            "type": "text",
            "text": "runtime input",
            "cache_control": {"type": "ephemeral"},
        }
    ]
