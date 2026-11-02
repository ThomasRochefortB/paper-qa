"""Exercise dataset conversion and splitting without downloading from the Hub."""

import pytest
from datasets import Dataset

from paperqa.litqa import read_litqa_v2_from_hub


@pytest.mark.parametrize("randomize", [False, True])
def test_litqa_hub_data_is_split_and_normalized_offline(monkeypatch, randomize):
    questions = [f"Question {index}" for index in range(5)]
    rows = Dataset.from_dict(
        {
            "question": questions,
            "ideal": ["Correct"] * 5,
            "distractors": [["Incorrect"]] * 5,
        }
    )
    calls = []

    def local_dataset(name, subset):
        calls.append((name, subset))
        return {"train": rows}

    monkeypatch.setattr("datasets.load_dataset", local_dataset)
    train, evaluation = read_litqa_v2_from_hub(
        "test/local", randomize=randomize, seed=42
    )
    assert calls == [("test/local", "LitQA2")]
    assert len(train) == 4
    assert len(evaluation) == 1
    combined = [*train.question, *evaluation.question]
    assert sorted(combined) == questions
    assert all(isinstance(value, list) for value in train.distractors)
    if not randomize:
        assert combined == questions
    else:
        repeat_train, repeat_eval = read_litqa_v2_from_hub("test/local", seed=42)
        assert [*repeat_train.question, *repeat_eval.question] == combined
