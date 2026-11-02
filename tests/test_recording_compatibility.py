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
    normalize_openai_stream_default,
)


def chat_request(payload, host="api.openai.com"):
    return Request(
        "POST",
        f"https://{host}/v1/chat/completions",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )


def test_non_streaming_default_matches_an_omitted_default() -> None:
    payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Q"}]}
    recorded = normalize_openai_stream_default(
        chat_request({**payload, "stream": False})
    )
    current = normalize_openai_stream_default(chat_request(payload))
    body(recorded, current)


@pytest.mark.parametrize(
    "changed", [{"model": "another-model"}, {"messages": []}, {"stream": True}]
)
def test_semantic_request_changes_still_fail_replay_matching(changed) -> None:
    payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Q"}]}
    recorded = normalize_openai_stream_default(
        chat_request({**payload, "stream": False})
    )
    current = normalize_openai_stream_default(chat_request({**payload, **changed}))
    with pytest.raises(AssertionError):
        body(recorded, current)


def test_other_hosts_are_not_normalized() -> None:
    request = chat_request({"stream": False}, host="example.org")
    original = request.body
    assert normalize_openai_stream_default(request).body == original


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
        before_record_request=normalize_openai_stream_default,
    )
    with recorder.use_cassette(str(path)) as cassette:
        assert cassette.can_play_response_for(chat_request(payload))
