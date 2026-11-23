from types import SimpleNamespace
import pytest
from fastapi import HTTPException
from app import security


def test_fail_closed_without_server_key(monkeypatch):
    monkeypatch.setattr(security, "get_settings", lambda: SimpleNamespace(api_key=""))
    with pytest.raises(HTTPException) as exc:
        security.require_api_key("anything")
    assert exc.value.status_code == 503


@pytest.mark.parametrize("value", [None, "", "incorrect"])
def test_reject_unauthorized(monkeypatch, value):
    monkeypatch.setattr(security, "get_settings", lambda: SimpleNamespace(api_key="secret"))
    with pytest.raises(HTTPException) as exc:
        security.require_api_key(value)
    assert exc.value.status_code == 401


def test_accept_authorized(monkeypatch):
    monkeypatch.setattr(security, "get_settings", lambda: SimpleNamespace(api_key="secret"))
    assert security.require_api_key("secret") is None
