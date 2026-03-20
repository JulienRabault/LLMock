"""Tests for all LLM provider routers added in MOC-6."""

import pytest
from fastapi.testclient import TestClient

from llmock.main import create_app

client = TestClient(create_app())


# ---------- Google Gemini ----------

def test_gemini_list_models():
    r = client.get("/gemini/v1beta/models")
    assert r.status_code == 200
    data = r.json()
    assert "models" in data
    assert len(data["models"]) > 0
    model = data["models"][0]
    assert model["name"].startswith("models/")


def test_gemini_generate_content():
    r = client.post(
        "/gemini/v1beta/models/gemini-2.0-flash:generateContent",
        json={
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}]
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "candidates" in data
    assert data["candidates"][0]["content"]["role"] == "model"
    assert "usageMetadata" in data


# ---------- Cohere ----------

def test_cohere_list_models():
    r = client.get("/cohere/v2/models")
    assert r.status_code == 200
    data = r.json()
    assert "models" in data
    assert any(m["name"] == "command-r-plus-08-2024" for m in data["models"])


def test_cohere_chat():
    r = client.post(
        "/cohere/v2/chat",
        json={
            "model": "command-r-plus",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["finish_reason"] == "COMPLETE"
    assert data["message"]["role"] == "assistant"
    assert len(data["message"]["content"]) > 0
    assert "usage" in data


# ---------- Groq ----------

def test_groq_list_models():
    r = client.get("/groq/openai/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    assert any(m["id"] == "llama-3.3-70b-versatile" for m in data["data"])


def test_groq_chat_completions():
    r = client.post(
        "/groq/openai/v1/chat/completions",
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"


def test_groq_rejects_n_greater_than_one():
    r = client.post(
        "/groq/openai/v1/chat/completions",
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": "Hello"}],
            "n": 2,
        },
    )
    assert r.status_code == 400
    assert "error" in r.json()


# ---------- Together AI ----------

def test_together_list_models():
    r = client.get("/together/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert len(data["data"]) > 0


def test_together_chat_completions():
    r = client.post(
        "/together/v1/chat/completions",
        json={
            "model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["choices"][0]["finish_reason"] == "eos"


# ---------- Perplexity AI ----------

def test_perplexity_list_models():
    r = client.get("/perplexity/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert any(m["id"] == "sonar-pro" for m in data["data"])


def test_perplexity_chat_completions():
    r = client.post(
        "/perplexity/v1/chat/completions",
        json={
            "model": "sonar-pro",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion"
    assert "citations" in data


def test_perplexity_optional_search_fields():
    r = client.post(
        "/perplexity/v1/chat/completions",
        json={
            "model": "sonar-pro",
            "messages": [{"role": "user", "content": "Hello"}],
            "return_citations": True,
            "return_images": True,
            "return_related_questions": True,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["search_results"]) == 1
    assert len(data["citations"]) == 1
    assert len(data["images"]) == 1
    assert data["images"][0]["image_url"]
    assert len(data["related_questions"]) == 2


# ---------- AI21 Labs ----------

def test_ai21_list_models():
    r = client.get("/ai21/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert any(m["id"] == "jamba-1.5-large" for m in data["data"])


def test_ai21_chat_completions():
    r = client.post(
        "/ai21/v1/chat/completions",
        json={
            "model": "jamba-1.5-large",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["choices"][0]["message"]["role"] == "assistant"


# ---------- xAI (Grok) ----------

def test_xai_list_models():
    r = client.get("/xai/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert any(m["id"] == "grok-3" for m in data["data"])


def test_xai_chat_completions():
    r = client.post(
        "/xai/v1/chat/completions",
        json={
            "model": "grok-3",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"]


def test_xai_vision_style_content_blocks():
    r = client.post(
        "/xai/v1/chat/completions",
        json={
            "model": "grok-2-vision-1212",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this mock image"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/mock.png"}},
                    ],
                }
            ],
        },
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"]
