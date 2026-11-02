"""Preserve numeric vectors across the patched SDK's response encodings."""

import base64
import struct
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from paperqa import LiteLLMEmbeddingModel, Settings, embedding_model_factory


@pytest.mark.asyncio
@pytest.mark.parametrize("encoded", [False, True])
async def test_embedding_responses_are_numeric_vectors(encoded: bool) -> None:
    values = [0.25, -1.0, 0.5]
    wire_embedding = (
        base64.b64encode(struct.pack("<3f", *values)).decode() if encoded else values
    )
    response = SimpleNamespace(data=[{"embedding": wire_embedding}])
    model = LiteLLMEmbeddingModel(name="text-embedding-3-small", ndim=3)
    with patch(
        "litellm.aembedding", new_callable=AsyncMock, return_value=response
    ) as embed:
        result = await model.embed_documents(["A document"])
    assert result == [values]
    embed.assert_awaited_once()


def test_settings_and_public_factory_use_the_compatible_embedding_adapter() -> None:
    assert isinstance(
        embedding_model_factory("text-embedding-3-small"), LiteLLMEmbeddingModel
    )
    assert isinstance(Settings().get_embedding_model(), LiteLLMEmbeddingModel)
