#!/usr/bin/env bash
# chaos_scenarios.sh — exercise LLMock across four chaos configurations.
#
# Each scenario starts LLMock with a different chaos profile, fires a few
# curl requests, then shuts it down.  Watch the HTTP status codes to confirm
# your client handles each failure mode correctly.
#
# Usage:
#   chmod +x examples/chaos_scenarios.sh
#   ./examples/chaos_scenarios.sh
#
# Requirements:
#   llmock installed  (pip install -e .)
#   curl, jq (optional, for pretty-printed output)

set -euo pipefail

HOST="127.0.0.1"
PORT="8765"
BASE="http://${HOST}:${PORT}"
LLMOCK_PID=""

# ── helpers ──────────────────────────────────────────────────────────────────

start_llmock() {
    local label="$1"; shift
    echo ""
    echo "════════════════════════════════════════════"
    echo "  Scenario: $label"
    echo "════════════════════════════════════════════"
    llmock serve --host "$HOST" --port "$PORT" "$@" &
    LLMOCK_PID=$!
    sleep 1   # let uvicorn start
}

stop_llmock() {
    if [[ -n "$LLMOCK_PID" ]]; then
        kill "$LLMOCK_PID" 2>/dev/null && wait "$LLMOCK_PID" 2>/dev/null || true
        LLMOCK_PID=""
    fi
}

trap stop_llmock EXIT

fire() {
    local label="$1"
    echo -n "  $label → "
    status=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "${BASE}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer test-key" \
        -d '{"model":"gpt-4o","messages":[{"role":"user","content":"ping"}]}')
    echo "HTTP $status"
}

fire_n() {
    local n="$1"
    local counts=""
    for i in $(seq 1 "$n"); do
        status=$(curl -s -o /dev/null -w "%{http_code}" \
            -X POST "${BASE}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer test-key" \
            -d '{"model":"gpt-4o","messages":[{"role":"user","content":"ping"}]}')
        counts="$counts $status"
    done
    echo "  10 requests:$counts"
}

# ── scenario 1: no chaos ──────────────────────────────────────────────────────

start_llmock "No chaos — all requests succeed"
fire "chat/completions"
fire "health check (GET /health)"  # should bypass chaos anyway
stop_llmock

# ── scenario 2: high latency ─────────────────────────────────────────────────

start_llmock "High latency (800 ms)" --latency-ms 800
echo -n "  Request (expect ~800 ms delay) → "
time curl -s -o /dev/null -w "HTTP %{http_code}\n" \
    -X POST "${BASE}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer test-key" \
    -d '{"model":"gpt-4o","messages":[{"role":"user","content":"ping"}]}'
stop_llmock

# ── scenario 3: 50 % rate-limit storm ────────────────────────────────────────

start_llmock "50% rate limits (429)" --error-rate 429=0.5
echo "  Firing 10 requests — expect ~5 to return 429:"
fire_n 10
stop_llmock

# ── scenario 4: cascading failures ───────────────────────────────────────────

start_llmock "Cascading failures (200ms latency + 20% 429 + 10% 500 + 10% 503)" \
    --latency-ms 200 \
    --error-rate 429=0.2 \
    --error-rate 500=0.1 \
    --error-rate 503=0.1
echo "  Firing 10 requests — expect a mix of 200, 429, 500, 503:"
fire_n 10
stop_llmock

echo ""
echo "Done.  Adjust error rates and latency to match your production failure patterns."
