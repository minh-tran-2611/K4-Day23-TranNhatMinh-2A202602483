import httpx
import pytest

import tools
from tools import RetryableError, with_retry


def flaky(failures, exc_factory):
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] <= failures:
            raise exc_factory()
        return "ok"
    return fn, calls


def test_retry_then_success_with_capped_backoff():
    sleeps = []
    fn, calls = flaky(3, lambda: RetryableError("busy"))
    assert with_retry(fn, attempts=5, base=1.0, cap=3.0, sleep=sleeps.append) == "ok"
    assert calls["n"] == 4
    assert len(sleeps) == 3 and all(0 < s <= 3.0 for s in sleeps)
    assert 1.0 <= sleeps[0] <= 2.0  # base * 2**0 plus jitter in [0, base]


def test_retry_after_is_respected_and_capped():
    sleeps = []
    fn, _ = flaky(2, lambda: RetryableError("429", retry_after=7))
    with_retry(fn, cap=5.0, sleep=sleeps.append)
    assert sleeps == [5.0, 5.0]


def test_gives_up_without_sleeping_after_last_attempt():
    sleeps = []
    fn, calls = flaky(10, lambda: RetryableError("down"))
    with pytest.raises(RetryableError):
        with_retry(fn, attempts=3, sleep=sleeps.append)
    assert calls["n"] == 3 and len(sleeps) == 2


def test_transport_errors_are_retried():
    fn, calls = flaky(1, lambda: httpx.ConnectTimeout("timeout"))
    assert with_retry(fn, sleep=lambda s: None) == "ok" and calls["n"] == 2


def test_other_errors_are_not_retried():
    fn, calls = flaky(1, lambda: ValueError("bug"))
    with pytest.raises(ValueError):
        with_retry(fn, sleep=lambda s: None)
    assert calls["n"] == 1


def test_arxiv_query_sanitised_without_network(monkeypatch):
    monkeypatch.setattr(tools, "_arxiv_get", lambda params: pytest.fail("network must not be called"))
    assert tools.arxiv_search.invoke({"query": "\"\" :: AND OR"}) == "NO RESULTS"


def test_exa_key_is_redacted(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "secret-key-123")

    def boom(url, **kwargs):
        raise httpx.HTTPStatusError(f"400 for {url}", request=None, response=None)
    monkeypatch.setattr(tools.httpx, "post", boom)
    out = tools.web_fetch.invoke({"url": "https://example.org"})
    assert out.startswith("ERROR:") and "secret-key-123" not in out
