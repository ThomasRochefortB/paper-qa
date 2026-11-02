"""Keep replay matching strict across equivalent SDK serialization defaults."""

import json

import pytest
import vcr
import yaml
from vcr.matchers import body
from vcr.request import Request

from .conftest import (
    CASSETTES_DIR,
    VCR_DEFAULT_MATCH_ON,
    normalize_openai_wire_defaults,
)


def api_request(payload, host="api.openai.com", path="chat/completions"):
    return Request(
        "POST",
        f"https://{host}/v1/{path}",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )


def test_non_streaming_default_matches_an_omitted_default() -> None:
    payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Q"}]}
    recorded = normalize_openai_wire_defaults(api_request(payload | {"stream": False}))
    current = normalize_openai_wire_defaults(api_request(payload))
    body(recorded, current)


@pytest.mark.parametrize(
    "changed", [{"model": "another-model"}, {"messages": []}, {"stream": True}]
)
def test_semantic_request_changes_still_fail_replay_matching(changed) -> None:
    payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Q"}]}
    recorded = normalize_openai_wire_defaults(api_request(payload | {"stream": False}))
    current = normalize_openai_wire_defaults(api_request(payload | changed))
    with pytest.raises(AssertionError):
        body(recorded, current)


def test_other_hosts_are_not_normalized() -> None:
    request = api_request({"stream": False}, host="example.org")
    original = request.body
    assert normalize_openai_wire_defaults(request).body == original


def test_existing_chat_recording_accepts_the_sdk_default_omission() -> None:
    path = (
        CASSETTES_DIR
        / "TestLitQAEvaluation.test_from_question[matched-correct-option].yaml"
    )
    recording = yaml.safe_load(path.read_text())
    payload = json.loads(recording["interactions"][0]["request"]["body"])
    assert payload.pop("stream") is False
    recorder = vcr.VCR(
        record_mode="none",
        match_on=[*VCR_DEFAULT_MATCH_ON, "body"],
        before_record_request=normalize_openai_wire_defaults,
    )
    with recorder.use_cassette(str(path)) as cassette:
        assert cassette.can_play_response_for(api_request(payload))


def test_embedding_encodings_match_as_equivalent_numeric_vectors() -> None:
    payload = {
        "model": "text-embedding-3-small",
        "input": ["A document"],
        "dimensions": 3,
    }
    recorded = normalize_openai_wire_defaults(
        api_request(payload | {"encoding_format": "base64"}, path="embeddings")
    )
    current = normalize_openai_wire_defaults(
        api_request(payload | {"encoding_format": "float"}, path="embeddings")
    )
    body(recorded, current)


@pytest.mark.parametrize(
    "changed",
    [
        {"model": "another-model"},
        {"input": ["Another document"]},
        {"dimensions": 6},
        {"encoding_format": "unsupported"},
    ],
)
def test_embedding_semantics_still_fail_replay_matching(changed) -> None:
    payload = {
        "model": "text-embedding-3-small",
        "input": ["A document"],
        "dimensions": 3,
        "encoding_format": "base64",
    }
    recorded = normalize_openai_wire_defaults(api_request(payload, path="embeddings"))
    current = normalize_openai_wire_defaults(
        api_request(payload | changed, path="embeddings")
    )
    with pytest.raises(AssertionError):
        body(recorded, current)


def test_existing_embedding_recording_accepts_float_wire_encoding() -> None:
    path = CASSETTES_DIR / "test_pdf_reader_match_doc_details.yaml"
    recording = yaml.safe_load(path.read_text())
    interaction = next(
        row
        for row in recording["interactions"]
        if row["request"]["uri"].endswith("/embeddings")
    )
    payload = json.loads(interaction["request"]["body"])
    assert payload["encoding_format"] == "base64"
    payload["encoding_format"] = "float"
    recorder = vcr.VCR(
        record_mode="none",
        match_on=[*VCR_DEFAULT_MATCH_ON, "body"],
        before_record_request=normalize_openai_wire_defaults,
    )
    with recorder.use_cassette(str(path)) as cassette:
        assert cassette.can_play_response_for(api_request(payload, path="embeddings"))
