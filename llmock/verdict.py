"""Judge a client's resilience from the journal.

LLMock sees every attempt and its timing, so it can tell not only that a
request failed, but how the client reacted: whether it waited as long as
``Retry-After`` asked, backed off, gave up on retryable errors, retried
errors that can never succeed, or quietly accepted a truncated stream.

Attempts are grouped into *calls*: a retry resends the same body, so an
attempt that repeats a failed one's fingerprint is its retry. SDK headers
such as ``x-stainless-retry-count`` are deliberately not used for this: they
only count retries made inside the SDK and read 0 on every attempt of an
application-level retry loop -- the very loops worth judging.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from llmock.journal import RequestRecord

__all__ = ["Call", "Finding", "Verdict", "judge"]

Severity = Literal["error", "warning"]

#: Client errors that no retry can fix.
NON_RETRYABLE = frozenset({400, 401, 403, 404, 422})
#: Errors worth retrying.
RETRYABLE = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
#: More attempts than this for a single call is a retry storm.
MAX_ATTEMPTS = 10
# Clocks are coarse on some platforms: forgive this much early retrying.
_SLACK_SECONDS = 0.05
_SLACK_RATIO = 0.05


@dataclass(frozen=True)
class Finding:
    severity: Severity
    code: str
    title: str
    advice: str
    call: int
    seqs: tuple[int, ...]
    path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "title": self.title,
            "advice": self.advice,
            "call": self.call,
            "requests": list(self.seqs),
            "path": self.path,
        }


@dataclass(frozen=True)
class Call:
    """One logical request and every attempt the client made at it."""

    attempts: tuple[RequestRecord, ...]

    @property
    def first(self) -> RequestRecord:
        return self.attempts[0]

    @property
    def last(self) -> RequestRecord:
        return self.attempts[-1]

    @property
    def succeeded(self) -> bool:
        return not _failed(self.last)


@dataclass(frozen=True)
class Verdict:
    calls: tuple[Call, ...]
    findings: tuple[Finding, ...] = field(default=())

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == "error")

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == "warning")

    @property
    def passed(self) -> bool:
        """No errors. Warnings do not fail a verdict unless asked to."""
        return not self.errors

    @property
    def attempts(self) -> int:
        return sum(len(c.attempts) for c in self.calls)

    @property
    def faults(self) -> int:
        return sum(1 for c in self.calls for a in c.attempts if a.fault)

    def assert_ok(self, *, strict: bool = False) -> None:
        """Raise ``AssertionError`` with the full report if the client misbehaved.

        ``strict`` fails on warnings too.
        """
        if self.errors or (strict and self.warnings):
            raise AssertionError("\n" + self.render())

    def render(self) -> str:
        head = (
            f"LLMock resilience verdict: {len(self.calls)} call(s), "
            f"{self.attempts} attempt(s), {self.faults} fault(s) injected"
        )
        if not self.findings:
            return f"{head}\n\n  PASS  the client handled every injected fault correctly.\n"
        lines = [head, ""]
        for f in self.findings:
            mark = "FAIL" if f.severity == "error" else "WARN"
            refs = ", ".join(f"#{s}" for s in f.seqs)
            lines.append(f"  {mark}  {f.code}  {f.path}  ({refs})")
            lines.append(f"        {f.title}")
            lines.append(f"        -> {f.advice}")
            lines.append("")
        lines.append(f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)")
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "calls": len(self.calls),
            "attempts": self.attempts,
            "faults_injected": self.faults,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "findings": [f.to_dict() for f in self.findings],
        }


def judge(records: Iterable[RequestRecord]) -> Verdict:
    calls = group_calls(records)
    findings: list[Finding] = []
    for index, call in enumerate(calls, start=1):
        for check in _CHECKS:
            findings.extend(check(call, index))
    order = {"error": 0, "warning": 1}
    findings.sort(key=lambda f: (order[f.severity], f.call, f.seqs))
    return Verdict(calls=tuple(calls), findings=tuple(findings))


def group_calls(records: Iterable[RequestRecord]) -> list[Call]:
    """Group attempts of the same logical call, in arrival order."""
    groups: list[list[RequestRecord]] = []
    open_by_fingerprint: dict[str, list[RequestRecord]] = {}
    for record in sorted(records, key=lambda r: r.seq):
        group = open_by_fingerprint.get(record.fingerprint)
        if group is not None:
            group.append(record)
        else:
            group = [record]
            groups.append(group)
        # Only a failed attempt can be followed by a retry.
        if _failed(record):
            open_by_fingerprint[record.fingerprint] = group
        else:
            open_by_fingerprint.pop(record.fingerprint, None)
    return [Call(tuple(g)) for g in groups]


def _failed(record: RequestRecord) -> bool:
    return record.status >= 400 or not record.completed


def _gap(before: RequestRecord, after: RequestRecord) -> float:
    """How long the client waited between the end of one attempt and the next."""
    return after.started_at - before.ended_at


def _finding(severity: Severity, code: str, title: str, advice: str, call: Call,
             index: int, *records: RequestRecord) -> Finding:
    picked = records or call.attempts
    return Finding(severity, code, title, advice, index, tuple(r.seq for r in picked),
                   f"{call.first.method} {call.first.path}")


# -- checks --------------------------------------------------------------------


def _retry_after_ignored(call: Call, index: int) -> list[Finding]:
    out = []
    for before, after in zip(call.attempts, call.attempts[1:]):
        wanted = before.retry_after
        if wanted is None or before.status < 400:
            continue
        waited = _gap(before, after)
        if waited + _SLACK_SECONDS + wanted * _SLACK_RATIO < wanted:
            out.append(_finding(
                "error", "retry_after_ignored",
                f"Retried {waited:.2f}s after a {before.status}, but Retry-After asked for {wanted:.2f}s.",
                "Wait for Retry-After (or retry-after-ms) before retrying: an earlier "
                "retry lands in the same rate-limit window and fails again.",
                call, index, before, after,
            ))
    return out


def _retried_non_retryable(call: Call, index: int) -> list[Finding]:
    for before, after in zip(call.attempts, call.attempts[1:]):
        if before.status in NON_RETRYABLE:
            return [_finding(
                "error", "retried_non_retryable",
                f"Retried a {before.status}: the same request will fail the same way.",
                "Only retry 408, 409, 429, 5xx and dropped connections. Surface 4xx "
                "client errors to the caller instead.",
                call, index, before, after,
            )]
    return []


def _retry_storm(call: Call, index: int) -> list[Finding]:
    if len(call.attempts) <= MAX_ATTEMPTS:
        return []
    return [_finding(
        "error", "retry_storm",
        f"{len(call.attempts)} attempts for a single call.",
        f"Cap retries (the OpenAI and Anthropic SDKs default to 2); beyond "
        f"{MAX_ATTEMPTS} you are amplifying the outage you are trying to survive.",
        call, index,
    )]


def _truncated_stream_accepted(call: Call, index: int) -> list[Finding]:
    last = call.last
    if last.fault is None or not last.fault.startswith("stream:truncate"):
        return []
    return [_finding(
        "error", "truncated_stream_accepted",
        f"A stream was cut after {last.chunks_sent} chunk(s) with no finish reason, "
        "and the client did not retry.",
        "Treat a stream that ends without a finish reason (or [DONE], or "
        "message_stop) as a failure: SDKs raise nothing, so the app probably "
        "used half an answer.",
        call, index, last,
    )]


def _malformed_chunk_ignored(call: Call, index: int) -> list[Finding]:
    last = call.last
    if last.fault is None or not last.fault.startswith("stream:malformed"):
        return []
    return [_finding(
        "warning", "malformed_chunk_ignored",
        "A malformed chunk was sent and the client neither retried nor stopped reading.",
        "Some SDKs skip undecodable events silently (Cohere) and others raise a raw "
        "JSONDecodeError: check that no text went missing.",
        call, index, last,
    )]


def _no_backoff(call: Call, index: int) -> list[Finding]:
    gaps = [
        _gap(b, a) for b, a in zip(call.attempts, call.attempts[1:])
        if _failed(b) and b.retry_after is None
    ]
    if len(gaps) < 2:
        return []
    if all(g < 0.05 for g in gaps):
        title = f"Retried {len(gaps)} times with no delay at all."
    elif all(later <= earlier * 1.1 for earlier, later in zip(gaps, gaps[1:])):
        title = f"Retried {len(gaps)} times at a constant interval (~{gaps[0]:.2f}s)."
    else:
        return []
    return [_finding(
        "warning", "no_backoff", title,
        "Back off exponentially, with jitter, when no Retry-After is given.",
        call, index,
    )]


def _gave_up(call: Call, index: int) -> list[Finding]:
    last = call.last
    retryable = last.status in RETRYABLE or (
        last.fault is not None and last.fault.startswith("stream:disconnect")
    )
    if call.succeeded or not retryable:
        return []
    what = "a dropped connection" if last.status < 400 else f"a {last.status}"
    return [_finding(
        "warning", "gave_up",
        f"Gave up after {what} without retrying ({len(call.attempts)} attempt(s)).",
        "Transient failures are retryable: allow a couple of retries with backoff, "
        "or make sure the caller handles the error on purpose.",
        call, index, last,
    )]


def _no_read_timeout(call: Call, index: int) -> list[Finding]:
    out = []
    for record in call.attempts:
        if record.stall_waited is not None and record.completed and record.stall_waited >= 1.0:
            out.append(_finding(
                "warning", "no_read_timeout",
                f"The client sat through a {record.stall_waited:.1f}s stall without timing out.",
                "Set a read timeout on streams so a hung connection fails fast and can be retried.",
                call, index, record,
            ))
    return out


_CHECKS = (
    _retry_after_ignored,
    _retried_non_retryable,
    _retry_storm,
    _truncated_stream_accepted,
    _malformed_chunk_ignored,
    _no_backoff,
    _gave_up,
    _no_read_timeout,
)
