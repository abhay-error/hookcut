"""Minimal JSON-mode chat client for local open models.

Two wire formats cover nearly every local runtime:

* ``ollama``  - Ollama's native /api/chat (supports schema-constrained output,
  context size, and thinking control).
* ``openai``  - the OpenAI-compatible /v1/chat/completions that LM Studio,
  llama.cpp's server, vLLM, Jan, and others expose.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from .config import LLMSettings

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class LLMError(RuntimeError):
    pass


def parse_json(text: str) -> Any:
    """Pull a JSON object out of model output that may include thinking or fences."""
    text = _THINK.sub("", text).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"Model returned invalid JSON: {exc}") from exc
    raise LLMError("Model returned no JSON object.")


def chat_json(settings: LLMSettings, system: str, user: str, schema: dict) -> Any:
    try:
        if settings.backend == "ollama":
            raw = _ollama(settings, system, user, schema)
        else:
            raw = _openai(settings, system, user, schema)
    except httpx.ConnectError as exc:
        raise LLMError(
            f"Could not reach the model server at {settings.base_url}. "
            + ("Is Ollama running? Start it with `ollama serve`." if settings.backend == "ollama"
               else "Is your local server running?")
        ) from exc
    except httpx.TimeoutException as exc:
        raise LLMError(
            "The model took too long. Try a smaller model or a shorter window."
        ) from exc
    return parse_json(raw)


def _ollama(settings: LLMSettings, system: str, user: str, schema: dict) -> str:
    payload = {
        "model": settings.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "format": schema,
        "think": settings.think,
        "options": {"temperature": settings.temperature, "num_ctx": settings.num_ctx},
    }
    url = settings.base_url.rstrip("/") + "/api/chat"
    with httpx.Client(timeout=httpx.Timeout(settings.timeout_s, connect=10)) as client:
        response = client.post(url, json=payload)
        if response.status_code == 400 and "think" in response.text.lower():
            # Model doesn't support the think flag; retry without it.
            payload.pop("think")
            response = client.post(url, json=payload)
    if response.status_code == 404:
        raise LLMError(
            f"Ollama has no model '{settings.model}'. Pull it: `ollama pull {settings.model}`."
        )
    if response.status_code >= 400:
        raise LLMError(f"Ollama error {response.status_code}: {response.text[:300]}")
    return response.json().get("message", {}).get("content", "")


def _openai(settings: LLMSettings, system: str, user: str, schema: dict) -> str:
    base = settings.base_url.rstrip("/")
    url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
    payload: dict[str, Any] = {
        "model": settings.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": settings.temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "clips", "schema": schema},
        },
    }
    with httpx.Client(timeout=httpx.Timeout(settings.timeout_s, connect=10)) as client:
        response = client.post(url, json=payload)
        if response.status_code == 400:
            # Some servers don't support schemas; the prompt still asks for JSON.
            payload.pop("response_format")
            response = client.post(url, json=payload)
    if response.status_code >= 400:
        raise LLMError(f"Server error {response.status_code}: {response.text[:300]}")
    return response.json()["choices"][0]["message"]["content"] or ""


def list_models(settings: LLMSettings) -> list[str]:
    """Installed models, for the UI picker. Empty list if the server is down."""
    try:
        with httpx.Client(timeout=5) as client:
            if settings.backend == "ollama":
                data = client.get(settings.base_url.rstrip("/") + "/api/tags").json()
                return sorted(m["name"] for m in data.get("models", []))
            base = settings.base_url.rstrip("/")
            data = client.get(base + ("/models" if base.endswith("/v1") else "/v1/models")).json()
            return sorted(m["id"] for m in data.get("data", []))
    except Exception:
        return []


# -- model management (Ollama only) --------------------------------------------

MODEL_NAME = re.compile(r"^[a-z0-9][a-z0-9._\-/]*(:[a-z0-9][a-z0-9._\-]*)?$")


def installed_models(settings: LLMSettings) -> list[dict]:
    """[{name, size_gb}] for Ollama; names only for other servers."""
    if settings.backend != "ollama":
        return [{"name": m, "size_gb": None} for m in list_models(settings)]
    try:
        with httpx.Client(timeout=5) as client:
            data = client.get(settings.base_url.rstrip("/") + "/api/tags").json()
    except Exception:
        return []
    return sorted(
        ({"name": m["name"], "size_gb": round(m.get("size", 0) / 1e9, 1)}
         for m in data.get("models", [])),
        key=lambda m: m["name"],
    )


def pull_model(settings: LLMSettings, name: str, on_progress) -> None:
    """Download a model through Ollama, reporting (status, fraction) as it goes."""
    if settings.backend != "ollama":
        raise LLMError("Downloading models is only supported with Ollama.")
    if not MODEL_NAME.match(name):
        raise LLMError(f"'{name}' is not a valid model name.")
    url = settings.base_url.rstrip("/") + "/api/pull"
    try:
        with httpx.Client(timeout=httpx.Timeout(60, connect=10, read=300)) as client, \
                client.stream("POST", url, json={"model": name, "stream": True}) as response:
            if response.status_code >= 400:
                raise LLMError(f"Ollama refused the download: {response.read().decode()[:200]}")
            for line in response.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if "error" in event:
                    raise LLMError(event["error"])
                total, done = event.get("total"), event.get("completed")
                on_progress(event.get("status", ""), (done / total) if total and done else None)
    except httpx.ConnectError as exc:
        raise LLMError("Ollama is not running. Start it with `ollama serve`.") from exc


def delete_model(settings: LLMSettings, name: str) -> None:
    if settings.backend != "ollama":
        raise LLMError("Removing models is only supported with Ollama.")
    with httpx.Client(timeout=30) as client:
        response = client.request("DELETE", settings.base_url.rstrip("/") + "/api/delete",
                                  json={"model": name})
    if response.status_code >= 400:
        raise LLMError(f"Could not remove {name}: {response.text[:200]}")
