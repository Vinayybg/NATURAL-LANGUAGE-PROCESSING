"""
agent/llm.py — Pluggable LLM backend routing (Groq / Ollama / HuggingFace).
All three use open-source models and are freely accessible.
"""

import os
import re
import logging
import requests
from nvidia_agent import config

log = logging.getLogger(__name__)


def _call_groq(prompt: str, system: str) -> str:
    if not config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set. See .env.example.")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {config.GROQ_API_KEY}",
                 "Content-Type": "application/json"},
        json={"model": config.GROQ_MODEL, "messages": messages,
              "temperature": config.LLM_TEMPERATURE, "max_tokens": config.LLM_MAX_TOKENS},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _call_ollama(prompt: str, system: str) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # OLLAMA_THINK=false disables qwen3's chain-of-thought thinking output.
    # Without this, qwen3 prepends long "Thinking..." blocks before the JSON
    # which breaks _parse_json() in ceo_agent.py.
    think_mode = getattr(config, "OLLAMA_THINK", False)

    r = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/chat",
        json={
            "model":    config.OLLAMA_MODEL,
            "messages": messages,
            "stream":   False,
            "think":    think_mode,
            "options":  {
                "temperature": config.LLM_TEMPERATURE,
                "num_predict": config.LLM_MAX_TOKENS,
            },
        },
        timeout=300,  # local models need more time than cloud APIs
    )
    r.raise_for_status()
    content = r.json()["message"]["content"].strip()

    # Strip any <think>...</think> blocks the model still emits
    # even when think=false (safety net)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content


def _call_huggingface(prompt: str, system: str) -> str:
    if not config.HF_API_KEY:
        raise ValueError("HF_API_KEY not set. See .env.example.")
    full = f"<s>[INST] {system}\n\n{prompt} [/INST]" if system else f"<s>[INST] {prompt} [/INST]"
    r = requests.post(
        f"https://api-inference.huggingface.co/models/{config.HF_MODEL}",
        headers={"Authorization": f"Bearer {config.HF_API_KEY}",
                 "Content-Type": "application/json"},
        json={"inputs": full,
              "parameters": {"max_new_tokens": config.LLM_MAX_TOKENS,
                             "temperature": config.LLM_TEMPERATURE,
                             "return_full_text": False}},
        timeout=120,
    )
    r.raise_for_status()
    result = r.json()
    return (result[0].get("generated_text", "") if isinstance(result, list) else str(result)).strip()


def call_llm(prompt: str, system: str = "") -> str:
    """
    Route to the configured LLM backend.
    CHAT_BACKEND in .env overrides LLM_BACKEND for the chat feature only.
    This lets you use Groq for fast chat while the engine runs on Ollama.

    Example .env setup:
      LLM_BACKEND=ollama    ← intelligence engine uses local model
      CHAT_BACKEND=groq     ← CEO Advisor chat uses Groq (fast)
    """
    # CHAT_BACKEND overrides LLM_BACKEND for the chat feature only
    backend = os.getenv("CHAT_BACKEND", config.LLM_BACKEND).lower().strip()
    log.debug(f"LLM call: backend={backend}, prompt_len={len(prompt)}")
    if backend == "groq":        return _call_groq(prompt, system)
    if backend == "ollama":      return _call_ollama(prompt, system)
    if backend == "huggingface": return _call_huggingface(prompt, system)
    raise ValueError(f"Unknown LLM_BACKEND: '{backend}'. Choose: groq | ollama | huggingface")


def llm_available() -> bool:
    """Health check — returns True if the LLM backend responds."""
    try:
        return bool(call_llm("Reply with the single word: OK", "You are a test assistant."))
    except Exception as e:
        log.warning(f"LLM health check failed: {e}")
        return False
