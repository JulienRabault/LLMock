"""Scenario queue: matching, consumption, composition."""

from __future__ import annotations

import threading

import pytest

from llmock.scenarios import (
    Delay,
    Fail,
    Match,
    Reply,
    RequestInfo,
    ScenarioQueue,
    SlowFirstToken,
    StreamFault,
    ToolCall,
    ToolFault,
)

OPENAI = RequestInfo(provider="openai", method="POST", path="/v1/chat/completions", model="gpt-4o")
OPENAI_STREAM = RequestInfo(
    provider="openai", method="POST", path="/v1/chat/completions", model="gpt-4o", stream=True
)
ANTHROPIC = RequestInfo(
    provider="anthropic", method="POST", path="/anthropic/v1/messages", model="claude-sonnet-4"
)


# -- consumption --------------------------------------------------------------


def test_empty_queue_gives_an_empty_plan():
    assert ScenarioQueue().plan_for(OPENAI).is_empty


def test_a_behaviour_fires_once_by_default():
    queue = ScenarioQueue()
    queue.add(Fail(429))
    assert queue.plan_for(OPENAI).fail == Fail(429)
    assert queue.plan_for(OPENAI).fail is None


def test_times_counts_down():
    queue = ScenarioQueue()
    queue.add(Fail(503, times=3))
    fired = [queue.plan_for(OPENAI).fail is not None for _ in range(5)]
    assert fired == [True, True, True, False, False]


def test_times_none_is_permanent_until_cleared():
    queue = ScenarioQueue()
    queue.add(Fail(503, times=None))
    assert all(queue.plan_for(OPENAI).fail for _ in range(50))
    queue.clear()
    assert queue.plan_for(OPENAI).fail is None


def test_behaviours_fire_in_fifo_order_within_a_category():
    queue = ScenarioQueue()
    queue.add(Fail(429), Fail(500), Fail(503))
    assert [queue.plan_for(OPENAI).fail.status for _ in range(3)] == [429, 500, 503]


def test_pending_lists_what_is_left():
    queue = ScenarioQueue()
    queue.add(Fail(429, times=2), Delay(0.1))
    queue.plan_for(OPENAI)  # consumes one 429; the Fail short-circuits, Delay stays
    assert queue.pending() == [Fail(429, times=2), Delay(0.1)]


# -- composition --------------------------------------------------------------


def test_categories_compose_on_the_same_request():
    queue = ScenarioQueue()
    call = ToolCall("get_weather", {"city": "Paris"})
    queue.add(Reply(tool_calls=(call,)), StreamFault("disconnect", after_chunks=3))
    plan = queue.plan_for(OPENAI_STREAM)
    assert plan.reply.tool_calls == (call,)
    assert plan.stream_fault.kind == "disconnect"


def test_a_fail_keeps_the_other_categories_for_the_next_request():
    """An error answer never reaches the handler, so a reply must not be wasted on it."""
    queue = ScenarioQueue()
    queue.add(Fail(429), Reply(text="recovered"))
    first = queue.plan_for(OPENAI)
    assert first.fail is not None and first.reply is None
    second = queue.plan_for(OPENAI)
    assert second.fail is None and second.reply.text == "recovered"


# -- matching -----------------------------------------------------------------


def test_match_by_provider():
    queue = ScenarioQueue()
    queue.add(Fail(529, match=Match(provider="anthropic")))
    assert queue.plan_for(OPENAI).fail is None
    assert queue.plan_for(ANTHROPIC).fail.status == 529


def test_non_matching_behaviour_stays_queued():
    queue = ScenarioQueue()
    queue.add(Fail(529, match=Match(provider="anthropic")))
    for _ in range(3):
        queue.plan_for(OPENAI)
    assert len(queue.pending()) == 1


def test_match_model_with_wildcard():
    queue = ScenarioQueue()
    queue.add(Fail(404, match=Match(model="gpt-4*")))
    assert queue.plan_for(ANTHROPIC).fail is None
    assert queue.plan_for(OPENAI).fail.status == 404


def test_match_path_with_wildcard():
    queue = ScenarioQueue()
    queue.add(Fail(500, match=Match(path="/v1/embeddings")))
    assert queue.plan_for(OPENAI).fail is None


def test_stream_faults_only_match_streaming_requests_by_default():
    queue = ScenarioQueue()
    queue.add(StreamFault("truncate"), SlowFirstToken(1.0))
    assert queue.plan_for(OPENAI).stream_fault is None
    plan = queue.plan_for(OPENAI_STREAM)
    assert plan.stream_fault.kind == "truncate"
    assert plan.slow_first_token.seconds == 1.0


# -- validation ---------------------------------------------------------------


@pytest.mark.parametrize("status", [200, 302, 399, 600])
def test_fail_rejects_non_error_statuses(status):
    with pytest.raises(ValueError):
        Fail(status)


@pytest.mark.parametrize("times", [0, -1])
def test_times_must_be_positive(times):
    with pytest.raises(ValueError):
        Fail(500, times=times)


def test_unknown_stream_fault_kind_is_rejected():
    with pytest.raises(ValueError):
        StreamFault("explode")


def test_unknown_tool_fault_kind_is_rejected():
    with pytest.raises(ValueError):
        ToolFault("hallucinate")


def test_negative_delays_are_rejected():
    with pytest.raises(ValueError):
        Delay(-1)
    with pytest.raises(ValueError):
        SlowFirstToken(-0.5)


def test_add_rejects_non_behaviours():
    with pytest.raises(TypeError):
        ScenarioQueue().add("fail please")


def test_behaviours_are_immutable():
    fail = Fail(429)
    with pytest.raises(AttributeError):
        fail.status = 500


# -- concurrency --------------------------------------------------------------


def test_a_single_shot_behaviour_fires_exactly_once_under_contention():
    queue = ScenarioQueue()
    queue.add(Fail(429, times=5))
    hits = []
    barrier = threading.Barrier(32)

    def worker():
        barrier.wait()
        if queue.plan_for(OPENAI).fail is not None:
            hits.append(1)

    threads = [threading.Thread(target=worker) for _ in range(32)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(hits) == 5
