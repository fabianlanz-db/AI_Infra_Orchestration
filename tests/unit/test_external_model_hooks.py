import pytest

from framework.external_model_hooks import _extract_by_path


def test_extract_by_path_dotted():
    payload = {"choices": [{"message": {"content": "hello"}}]}
    assert _extract_by_path(payload, "choices.0.message.content") == "hello"


def test_extract_by_path_flat_key():
    assert _extract_by_path({"text": "hi"}, "text") == "hi"


def test_extract_by_path_missing_key_raises():
    with pytest.raises(KeyError):
        _extract_by_path({"a": 1}, "b")


def test_extract_by_path_invalid_index_raises():
    with pytest.raises(IndexError):
        _extract_by_path({"a": [1]}, "a.5")
