from app import stream_summary, summarize


def test_summary(llmock):
    assert summarize("Q3 report")


def test_summary_under_rate_limit(llmock):
    llmock.rate_limit(retry_after=1)
    assert summarize("Q3 report")


def test_streamed_summary(llmock):
    llmock.truncate(after_chunks=3)
    assert stream_summary("Q3 report")
