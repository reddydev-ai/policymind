"""Opt-in evaluation evidence endpoint requires independent authorization."""
from types import SimpleNamespace
import pytest
from fastapi import HTTPException
from app import security


@pytest.fixture
def policy_settings(monkeypatch):
    state = SimpleNamespace(api_key="different-business-key", eval_api_key="evaluation-secret", enable_eval_api=True)
    monkeypatch.setattr(security, "get_settings", lambda: state)
    return state


def test_eval_endpoint_disabled_by_default(policy_settings):
    policy_settings.enable_eval_api = False
    with pytest.raises(HTTPException) as ex:
        security.require_eval_key("evaluation-secret")
    assert ex.value.status_code == 404


def test_eval_endpoint_fails_closed_if_secret_unconfigured(policy_settings):
    policy_settings.eval_api_key = ""
    with pytest.raises(HTTPException) as ex:
        security.require_eval_key("evaluation-secret")
    assert ex.value.status_code == 503


@pytest.mark.parametrize("key", [None, "", "different-business-key", "incorrect"])
def test_eval_key_is_independent(policy_settings, key):
    with pytest.raises(HTTPException) as ex:
        security.require_eval_key(key)
    assert ex.value.status_code == 403


def test_correct_eval_key_allows_evidence_access(policy_settings):
    assert security.require_eval_key("evaluation-secret") is None
