"""Offline coverage of the client APIs affected by the security refresh."""

from unittest.mock import AsyncMock, patch

import litellm
import pytest
from llmclient import LiteLLMModel

from paperqa import Docs, Settings


def test_default_settings_still_construct_the_client_and_router() -> None:
    model = Settings().get_llm()
    assert isinstance(model, LiteLLMModel)
    with patch.dict("os.environ", {"OPENAI_API_KEY": "offline-test-key"}):
        assert isinstance(model.router, litellm.Router)
    assert not Docs().texts


@pytest.mark.asyncio
async def test_chat_result_survives_the_client_sdk_upgrade() -> None:
    model = LiteLLMModel(name="gpt-4o-mini", llm_type="chat")
    response = litellm.ModelResponse(
        model="gpt-4o-mini",
        choices=[{"message": {"role": "assistant", "content": "A safe answer."}}],
        usage={"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
    )
    with patch.dict("os.environ", {"OPENAI_API_KEY": "offline-test-key"}):
        router = model.router
    with (
        patch.object(
            router, "acompletion", new_callable=AsyncMock, return_value=response
        ) as completion,
        patch.object(LiteLLMModel, "count_tokens", return_value=1),
    ):
        result = await model.run_prompt("Answer {question}", {"question": "safely"})
    assert result.text == "A safe answer."
    completion.assert_awaited_once()
