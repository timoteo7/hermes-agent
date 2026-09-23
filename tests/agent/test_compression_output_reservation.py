"""Internal output reservation for the single context-summary call."""

from unittest.mock import ANY, patch

import pytest

from agent.auxiliary_client import _build_call_kwargs
from agent.context_compressor import ContextCompressor


def test_lean_summary_reserves_headroom_for_the_full_prompt_target():
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        compressor = ContextCompressor(model="main", provider="openai-codex", quiet_mode=True)
    compressor.tail_mode = "lean"

    with (
        patch.object(compressor, "_compute_summary_budget", return_value=10_000),
        patch.object(compressor, "_call_summary_llm", return_value="summary") as call,
    ):
        compressor._generate_summary([{"role": "user", "content": "hello"}])

    assert call.call_args.args == (ANY, ANY, 18_200)


@pytest.mark.parametrize(
    "provider,model,base_url,expected_key",
    [
        ("custom:OpenAdapter", "GPT-5.6-Luna", "https://api.openadapter.in/v1", "max_completion_tokens"),
        ("openai-codex", "gpt-5.6-sol-900k", "https://chatgpt.com/backend-api/codex", "max_completion_tokens"),
        ("anthropic", "claude-sonnet-4-6", "https://api.anthropic.com/v1", None),
        ("nvidia", "moonshotai/kimi-k3", "https://integrate.api.nvidia.com/v1", None),
        ("bedrock", "anthropic.claude-sonnet-4", "https://bedrock-runtime.us-east-1.amazonaws.com", None),
        ("gemini", "gemini-2.5-pro", "https://generativelanguage.googleapis.com", None),
    ],
)
def test_compression_reservation_is_route_aware(provider, model, base_url, expected_key):
    kwargs = _build_call_kwargs(
        provider,
        model,
        [{"role": "user", "content": "summarize"}],
        max_tokens=18_200,
        base_url=base_url,
        task="compression",
    )

    token_keys = {"max_tokens", "max_completion_tokens", "max_output_tokens"} & kwargs.keys()
    if expected_key is None:
        assert not token_keys
    else:
        assert token_keys == {expected_key}
        assert kwargs[expected_key] == 18_200
