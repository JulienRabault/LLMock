import time

import openai


def summarize(text):
    client = openai.OpenAI(max_retries=0)
    for _ in range(3):
        try:
            reply = client.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": f"Summarize: {text}"}])
            return reply.choices[0].message.content
        except openai.RateLimitError:
            time.sleep(0.2)  # good enough?


def stream_summary(text):
    client = openai.OpenAI()
    summary = ""
    with client.chat.completions.create(
            model="gpt-4o", stream=True,
            messages=[{"role": "user", "content": f"Summarize: {text}"}]) as stream:
        for chunk in stream:
            summary += chunk.choices[0].delta.content or ""
    return summary
