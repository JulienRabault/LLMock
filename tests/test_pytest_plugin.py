"""The llmock fixture, exercised the way users use it: through pytester runs."""

import pytest

pytest.importorskip("openai")

# Application code under test: builds its client from the environment, with
# no idea LLMock exists. That is the whole point of the fixture.
APP = '''
import time
import openai

def ask(prompt, *, retries=2):
    client = openai.OpenAI(max_retries=retries)
    reply = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}])
    return reply.choices[0].message.content

def naive_ask(prompt):
    client = openai.OpenAI(max_retries=0)
    for _ in range(3):
        try:
            return client.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": prompt}]
            ).choices[0].message.content
        except openai.RateLimitError:
            time.sleep(0.05)  # ignores Retry-After
'''


@pytest.fixture
def project(pytester):
    pytester.makepyfile(app=APP)
    return pytester


def run(pytester, test_source, *args):
    pytester.makepyfile(test_it=test_source)
    return pytester.runpytest_subprocess("-p", "no:cacheprovider", *args)


def test_unchanged_app_code_is_pointed_at_llmock(project):
    result = run(project, '''
import app

def test_it(llmock):
    assert app.ask("hi")
    assert llmock.last_request.path == "/v1/chat/completions"
''')
    result.assert_outcomes(passed=1)


def test_real_api_keys_are_replaced(project, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key-that-must-not-be-used")
    result = run(project, '''
import os

def test_it(llmock):
    assert os.environ["OPENAI_API_KEY"] != "sk-real-key-that-must-not-be-used"
''')
    result.assert_outcomes(passed=1)


def test_a_well_behaved_client_passes_the_verdict(project):
    result = run(project, '''
import app

def test_it(llmock):
    llmock.rate_limit(times=2, retry_after=0.1)
    assert app.ask("hi")
    assert [r.status for r in llmock.requests] == [429, 429, 200]
    llmock.assert_resilient()
''')
    result.assert_outcomes(passed=1)


def test_a_naive_client_fails_with_the_report(project):
    result = run(project, '''
import app

def test_it(llmock):
    llmock.rate_limit(retry_after=1.0)
    assert app.naive_ask("hi")
    llmock.assert_resilient()
''')
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*retry_after_ignored*", "*Retry-After asked for 1.00s*"])


def test_each_test_starts_clean(project):
    result = run(project, '''
import app

def test_a_breaks_everything(llmock):
    llmock.outage()
    llmock.tool_mode("off")

def test_b_sees_a_healthy_server(llmock):
    assert app.ask("hi")
    assert len(llmock.requests) == 1
''')
    result.assert_outcomes(passed=2)


def test_stream_faults_and_scripted_replies(project):
    result = run(project, '''
import openai

def test_it(llmock):
    llmock.reply("Once upon a time").truncate(after_chunks=2)
    client = openai.OpenAI()
    text = ""
    with client.chat.completions.create(model="gpt-4o", stream=True,
            messages=[{"role": "user", "content": "story"}]) as stream:
        for chunk in stream:
            text += chunk.choices[0].delta.content or ""
    assert text == "Once "
    assert [f.code for f in llmock.verdict().errors] == ["truncated_stream_accepted"]
''')
    result.assert_outcomes(passed=1)


def test_scripted_tool_call(project):
    result = run(project, '''
import json, openai

def test_it(llmock):
    llmock.call_tool("get_weather", {"city": "Paris"})
    message = openai.OpenAI().chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "weather?"}]).choices[0].message
    call = message.tool_calls[0]
    assert (call.function.name, json.loads(call.function.arguments)) == ("get_weather", {"city": "Paris"})
''')
    result.assert_outcomes(passed=1)


def test_report_option_lists_misbehaving_tests(project):
    result = run(project, '''
import app

def test_bad(llmock):
    llmock.rate_limit(retry_after=1.0)
    app.naive_ask("hi")

def test_good(llmock):
    app.ask("hi")
''', "--llmock-report")
    result.assert_outcomes(passed=2)
    result.stdout.fnmatch_lines(["*LLMock resilience report*", "*test_it.py::test_bad*",
                                 "*retry_after_ignored*"])
    assert "test_it.py::test_good" not in result.stdout.str()


def test_other_sdks_are_redirected_too(project):
    pytest.importorskip("anthropic")
    pytest.importorskip("cohere")
    pytest.importorskip("google.genai")
    result = run(project, '''
import anthropic, cohere
from google import genai

def test_it(llmock):
    anthropic.Anthropic().messages.create(
        model="claude-sonnet-4-6", max_tokens=64, messages=[{"role": "user", "content": "hi"}])
    cohere.ClientV2().chat(model="command-r-plus", messages=[{"role": "user", "content": "hi"}])
    gemini = genai.Client()  # keep a reference: google-genai closes its HTTP client on GC
    gemini.models.generate_content(model="gemini-2.5-pro", contents="hi")
    assert [r.provider for r in llmock.requests] == ["anthropic", "cohere", "gemini"]
''')
    result.assert_outcomes(passed=1)
