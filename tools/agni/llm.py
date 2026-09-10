"""Agni LLM — unified inference backend for local and frontier models.

Supports three backends:
  - ollama:   Local Ollama instance (default)
  - anthropic: Anthropic Claude API (Messages API)
  - openai:   Any OpenAI-compatible endpoint (Together, Groq, vLLM, etc.)

Configuration via environment variables:
  AGNI_LLM_BACKEND   — "ollama" (default), "anthropic", or "openai"
  AGNI_LLM_MODEL     — model name (backend-specific)
  AGNI_LLM_URL       — endpoint URL (ollama/openai only)
  AGNI_LLM_API_KEY   — API key (anthropic/openai only)
  AGNI_LLM_MAX_TOKENS — max response tokens (default 512)

Per-module model override: set BOUNTY_MODEL, WOO_MODEL, AP_MODEL etc.
to override the default for specific subsystems. The module-level env var
takes precedence over AGNI_LLM_MODEL.
"""

import json
import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

BACKEND = os.environ.get("AGNI_LLM_BACKEND", "ollama")
DEFAULT_MODEL = os.environ.get("AGNI_LLM_MODEL", "mistral:7b")
DEFAULT_URL = os.environ.get("AGNI_LLM_URL", "http://127.0.0.1:11434")
API_KEY = os.environ.get("AGNI_LLM_API_KEY", "")
MAX_TOKENS = int(os.environ.get("AGNI_LLM_MAX_TOKENS", "512"))

CLAUDE_CREDS_PATH = os.path.expanduser("~/.claude/.credentials.json")


def _load_oauth_token() -> str:
    """Load OAuth token from Claude CLI credentials if available."""
    try:
        with open(CLAUDE_CREDS_PATH) as f:
            creds = json.load(f)
        return creds.get("claudeAiOauth", {}).get("accessToken", "")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return ""


def infer(
    prompt: str,
    system: str = "",
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: float = 0.1,
    timeout: int = 120,
    backend: Optional[str] = None,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[str]:
    """Run inference on any supported backend. Returns response text or None."""
    backend = backend or BACKEND
    model = model or DEFAULT_MODEL
    max_tokens = max_tokens or MAX_TOKENS
    url = url or DEFAULT_URL
    api_key = api_key or API_KEY

    try:
        if backend == "ollama":
            return _ollama(prompt, system, model, max_tokens, temperature, timeout, url)
        elif backend == "anthropic":
            return _anthropic(prompt, system, model, max_tokens, temperature, timeout, api_key)
        elif backend == "openai":
            return _openai(prompt, system, model, max_tokens, temperature, timeout, url, api_key)
        else:
            logger.error(f"Unknown backend: {backend}")
            return None
    except Exception as e:
        logger.error(f"LLM inference failed ({backend}/{model}): {e}")
        return None


def _ollama(prompt, system, model, max_tokens, temperature, timeout, url):
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    if system:
        body["system"] = system
    resp = requests.post(f"{url}/api/generate", json=body, timeout=timeout)
    if resp.status_code == 200:
        j = resp.json()
        out = j.get("response", "") or ""
        think = j.get("thinking", "") or ""
        if think and not out.strip():
            return think
        if think:
            return think + chr(10) + chr(10) + out
        return out
    logger.warning(f"Ollama {resp.status_code}: {resp.text[:200]}")
    return None


def _anthropic(prompt, system, model, max_tokens, temperature, timeout, api_key):
    if not api_key:
        api_key = _load_oauth_token()
    if not api_key:
        logger.error("No API key: set AGNI_LLM_API_KEY or log into Claude CLI")
        return None
    if api_key.startswith("sk-ant-oat"):
        headers = {
            "authorization": f"Bearer {api_key}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
    else:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
    messages = [{"role": "user", "content": prompt}]
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    if system:
        body["system"] = system
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers, json=body, timeout=timeout,
    )
    # Claude 5-series rejects `temperature` outright ("deprecated for this model"),
    # so a body that is valid for 4-series 400s on 5-series. Retry once without it
    # rather than hardcoding a model allowlist that will rot the same way. 2026-08-19.
    if resp.status_code == 400 and "temperature" in resp.text:
        body.pop("temperature", None)
        logger.info("retrying without temperature (5-series model)")
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers, json=body, timeout=timeout,
        )
    if resp.status_code == 200:
        data = resp.json()
        return "".join(b.get("text", "") for b in data.get("content", []))
    # 429 here is contention, not quota: a bare POST loses every race against clients
    # that retry. Documented 2026-08-14, unfixed until now. Bounded backoff, no jitter
    # storm -- 3 attempts, 2s/6s/14s.
    attempt = 0
    while resp.status_code == 429 and attempt < 3:
        wait = 2 + attempt * 4 + attempt * attempt * 2
        logger.info("Anthropic 429, retry %d in %ds" % (attempt + 1, wait))
        time.sleep(wait)
        attempt += 1
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers, json=body, timeout=timeout,
        )
    if resp.status_code == 200:
        data = resp.json()
        return "".join(b.get("text", "") for b in data.get("content", []))
    logger.warning(f"Anthropic {resp.status_code}: {resp.text[:200]}")
    return None


def _openai(prompt, system, model, max_tokens, temperature, timeout, url, api_key):
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    endpoint = url.rstrip("/")
    if not endpoint.endswith("/v1/chat/completions"):
        endpoint = f"{endpoint}/v1/chat/completions"
    resp = requests.post(endpoint, headers=headers, json=body, timeout=timeout)
    if resp.status_code == 200:
        data = resp.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
    logger.warning(f"OpenAI-compat {resp.status_code}: {resp.text[:200]}")
    return None
