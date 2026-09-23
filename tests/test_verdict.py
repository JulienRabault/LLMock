"""The resilience verdict: rules on synthetic journals, then real clients."""

import time

import httpx
import pytest

from llmock.journal import RequestRecord
from llmock.scenarios import Fail, StreamFault
from llmock.testing import LLMockServer
from llmock.verdict import group_calls, judge

_seq = iter(range(1, 10_000))


def attempt(start, *, status=200, retry_after=None, duration=0.01, fingerprint="call-a",
            fault=None, completed=True, retry_count=None, chunks=0, stall_waited=None):
    return RequestRecord(
        seq=next(_seq), provider="openai", method="POST", path="/v1/chat/completions",
        started_at=start, ended_at=start + duration, status=status, fingerprint=fingerprint,
        fault=fault, retry_after=retry_after, completed=completed, sdk_retry_count=retry_count,
        chunks_sent=chunks, stall_waited=stall_waited,
    )


def codes(verdict):
    return [f.code for f in verdict.findings]


# -- grouping -----------------------------------------------------------------


def test_a_retry_after_a_failure_joins_the_same_call():
    calls = group_calls([attempt(0, status=503), attempt(1)])
    assert len(calls) == 1 and len(calls[0].attempts) == 2


def test_an_identical_request_after_a_success_is_a_new_call():
    assert len(group_calls([attempt(0), attempt(1)])) == 2


def test_different_bodies_are_different_calls():
    assert len(group_calls([attempt(0, status=503), attempt(1, fingerprint="call-b")])) == 2


def test_application_level_retries_are_grouped_despite_the_sdk_header():
    """An app loop calling create() again sends x-stainless-retry-count: 0 each time.

    Those are still retries -- the ones most worth judging.
    """
    calls = group_calls([attempt(0, status=429, retry_count=0), attempt(1, retry_count=0)])
    assert len(calls) == 1


# -- rules --------------------------------------------------------------------


def test_clean_run_passes():
    verdict = judge([attempt(0), attempt(1, fingerprint="b")])
    assert verdict.passed and not verdict.findings
    assert "PASS" in verdict.render()


def test_waiting_for_retry_after_passes():
    verdict = judge([attempt(0, status=429, retry_after=2.0), attempt(2.05)])
    assert verdict.passed and verdict.findings == ()


def test_retrying_early_ignores_retry_after():
    verdict = judge([attempt(0, status=429, retry_after=2.0), attempt(0.3)])
    assert codes(verdict) == ["retry_after_ignored"]
    assert not verdict.passed
    assert "Retry-After asked for 2.00s" in verdict.render()


def test_retrying_a_client_error_is_flagged():
    verdict = judge([attempt(0, status=401), attempt(1)])
    assert "retried_non_retryable" in codes(verdict)


def test_a_retry_storm_is_flagged():
    records = [attempt(i * 0.5, status=500) for i in range(12)]
    assert "retry_storm" in codes(judge(records))


def test_constant_interval_retries_lack_backoff():
    records = [attempt(t, status=500) for t in (0, 1, 2, 3)] + [attempt(4)]
    assert "no_backoff" in codes(judge(records))


def test_immediate_retries_lack_backoff():
    records = [attempt(t, status=500, duration=0.001) for t in (0, 0.002, 0.004)] + [attempt(0.006)]
    verdict = judge(records)
    finding = next(f for f in verdict.findings if f.code == "no_backoff")
    assert "no delay" in finding.title


def test_exponential_backoff_is_fine():
    records = [attempt(t, status=500) for t in (0, 0.5, 1.5, 3.5)] + [attempt(7.5)]
    assert "no_backoff" not in codes(judge(records))


def test_accepting_a_truncated_stream_is_an_error():
    verdict = judge([attempt(0, fault="stream:truncate@3", completed=False, chunks=3)])
    assert codes(verdict) == ["truncated_stream_accepted"]
    assert "3 chunk(s)" in verdict.findings[0].title


def test_retrying_a_truncated_stream_is_fine():
    verdict = judge([attempt(0, fault="stream:truncate@3", completed=False), attempt(1)])
    assert verdict.passed


def test_giving_up_on_a_retryable_error_is_a_warning():
    verdict = judge([attempt(0, status=503)])
    assert codes(verdict) == ["gave_up"]
    assert verdict.passed  # a warning does not fail the verdict...


def test_strict_mode_fails_on_warnings():
    with pytest.raises(AssertionError, match="gave_up"):
        judge([attempt(0, status=503)]).assert_ok(strict=True)


def test_giving_up_on_a_client_error_is_normal():
    assert judge([attempt(0, status=400)]).findings == ()


def test_waiting_out_a_whole_stall_means_no_read_timeout():
    verdict = judge([attempt(0, fault="stream:stall@2", duration=30, stall_waited=30.0)])
    assert "no_read_timeout" in codes(verdict)


def test_assert_ok_raises_with_the_report():
    verdict = judge([attempt(0, status=429, retry_after=2.0), attempt(0.1)])
    with pytest.raises(AssertionError, match="retry_after_ignored"):
        verdict.assert_ok()


def test_to_dict_is_json_ready():
    data = judge([attempt(0, status=401), attempt(1)]).to_dict()
    assert data["passed"] is False
    assert data["findings"][0]["code"] == "retried_non_retryable"


# -- real clients -------------------------------------------------------------


@pytest.fixture(scope="module")
def server():
    with LLMockServer() as running:
        yield running


@pytest.fixture(autouse=True)
def clean(server):
    server.state.reset()
    yield


CHAT = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}


def verdict_of(server):
    return judge(server.state.journal.records(wait=2))


def test_the_openai_sdk_honours_retry_after(server):
    openai = pytest.importorskip("openai")
    server.state.scenarios.add(Fail(429, retry_after=0.4, times=2))
    client = openai.OpenAI(base_url=server.base_url("openai"), api_key="t", max_retries=2)
    client.chat.completions.create(**CHAT)
    verdict = verdict_of(server)
    assert verdict.passed, verdict.render()
    assert verdict.attempts == 3


def test_a_naive_retry_loop_is_caught(server):
    """The loop everybody has written once: catch, sleep a bit, try again."""
    server.state.scenarios.add(Fail(429, retry_after=2.0))
    url = server.base_url("openai") + "/chat/completions"
    for _ in range(3):
        if httpx.post(url, json=CHAT).status_code == 200:
            break
        time.sleep(0.1)
    verdict = verdict_of(server)
    assert [f.code for f in verdict.errors] == ["retry_after_ignored"]


def test_retrying_a_401_is_caught(server):
    server.state.scenarios.add(Fail(401))
    url = server.base_url("openai") + "/chat/completions"
    for _ in range(2):
        httpx.post(url, json=CHAT)
    assert "retried_non_retryable" in [f.code for f in verdict_of(server).errors]


def test_the_openai_sdk_accepts_a_truncated_stream(server):
    """The silent bug LLMock exists to reveal."""
    openai = pytest.importorskip("openai")
    server.state.scenarios.add(StreamFault("truncate", after_chunks=3))
    client = openai.OpenAI(base_url=server.base_url("openai"), api_key="t", max_retries=2)
    with client.chat.completions.create(**CHAT, stream=True) as stream:
        for _ in stream:
            pass
    assert [f.code for f in verdict_of(server).errors] == ["truncated_stream_accepted"]


def test_no_retries_on_a_503_is_a_warning(server):
    openai = pytest.importorskip("openai")
    server.state.scenarios.add(Fail(503))
    client = openai.OpenAI(base_url=server.base_url("openai"), api_key="t", max_retries=0)
    with pytest.raises(openai.InternalServerError):
        client.chat.completions.create(**CHAT)
    assert [f.code for f in verdict_of(server).findings] == ["gave_up"]


# -- concurrency (from code review) ----------------------------------------------


def test_concurrent_identical_requests_are_separate_calls():
    """asyncio.gather of the same prompt, all rate-limited: 12 calls, not 12 retries."""
    records = [attempt(0.0 + i * 0.001, status=429, duration=0.2, retry_after=1.0) for i in range(12)]
    verdict = judge(records)
    assert len(verdict.calls) == 12
    assert "retry_storm" not in codes(verdict)
    assert "retry_after_ignored" not in codes(verdict)


def test_a_retry_starts_after_its_failure_ended():
    """Overlapping in time rules out a retry; following it does not."""
    calls = group_calls([
        attempt(0.0, status=503, duration=0.5),
        attempt(0.2, status=503, duration=0.5),   # overlaps the first: another call
        attempt(1.0),                              # after both: a retry
    ])
    assert sorted(len(c.attempts) for c in calls) == [1, 2]


def test_parallel_calls_each_keep_their_own_retries():
    records = [
        attempt(0.0, status=429, duration=0.1, retry_after=0.5),
        attempt(0.01, status=429, duration=0.1, retry_after=0.5),
        attempt(0.7), attempt(0.71),
    ]
    verdict = judge(records)
    assert [len(c.attempts) for c in verdict.calls] == [2, 2]
    assert verdict.passed


def test_retrying_then_giving_up_is_correct_behaviour():
    """Bounded retries that all fail are what a good client does during an outage."""
    records = [attempt(t, status=503, retry_after=0.5) for t in (0, 1, 2, 3)]
    assert "gave_up" not in codes(judge(records))
