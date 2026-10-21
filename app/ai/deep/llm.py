"""A small, self-contained Azure client for the deep engine.

Deliberately does NOT import from orchestrator.py. Deep mode is a second, heavier answer
path that must never be able to break the fast one — sharing a module means a change made
for the reasoning loop can regress the two-second path that answers most questions. The
duplication here is about forty lines and it buys complete isolation.
"""
from __future__ import annotations

import functools
import json
import os
import time

from app.ai import windows

ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://ed-gpt.openai.azure.com")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
# The fast path reads AZURE_OPENAI_MINI_DEPLOYMENT and this module used to read
# AZURE_OPENAI_DEPLOYMENT_MINI, so setting one silently left the other on the default name —
# which may not exist on the Azure resource. Both spellings are honoured, the fast path's first.
MINI = (os.getenv("AZURE_OPENAI_MINI_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT_MINI")
        or "gpt-4o-mini")

# Which model runs which role.
#
# NOT a cost split. 4o-mini measured SLOWER than 4o in this application (2.88s vs 2.20s
# per call), so it buys no latency, and putting it on SQL generation would raise the error
# rate on the exact step that fabricates.
#
# It earns its place on ONE role: corroboration. Re-asking the same model to check its own
# figure mostly reproduces its own premise — that is the failure mode where five agents
# agree beautifully and are all wrong together. A DIFFERENT model deriving the same number
# by a different route is genuine independence, and it is the difference between
# verification and echo. Everything judgement-heavy stays on 4o.
ROLE_MODEL = {
    "frame":       DEPLOYMENT,
    "plan":        DEPLOYMENT,
    "sql":         DEPLOYMENT,   # highest-risk step — never downgrade
    "corroborate": MINI,         # independence, deliberately a different model
    "critique":    DEPLOYMENT,
    "gaps":        MINI,         # cheap, high-volume, and its misses are caught by the critic
    "synthesise":  DEPLOYMENT,
}


def model_for(role: str) -> str:
    return ROLE_MODEL.get(role, DEPLOYMENT)

_RETRIABLE = {429, 500, 502, 503, 504}
_MAX_RETRIES = 4


def has_key() -> bool:
    return bool(os.getenv("AZURE_OPENAI_API_KEY"))


def client():
    key = os.getenv("AZURE_OPENAI_API_KEY")
    if not key:
        raise RuntimeError("AZURE_OPENAI_API_KEY is not set")
    return _client_for(ENDPOINT, key, API_VERSION, float(os.getenv("AZURE_OPENAI_TIMEOUT", "60")))


@functools.lru_cache(maxsize=4)
def _client_for(endpoint: str, key: str, version: str, timeout: float):
    """One client per configuration, reused across questions.

    A new AzureOpenAI per question meant a new connection pool, so every deep question paid
    a fresh TCP and TLS handshake to Azure before its first token. The client is
    thread-safe, and the key is part of the cache key, so rotating it still takes effect.

    An explicit timeout, because the default is effectively "wait". A 90-case eval run
    stalled for 105 minutes at 0% CPU on a single request that never returned — and the
    same stall in production is a chat that spins forever with nothing to show the user.
    httpx applies the read timeout PER CHUNK, so a long stream is unaffected; only silence
    is. 60 s of silence is already a dead call.

    max_retries=0: chat() below is the ONE retry layer. With the SDK's own two retries
    underneath it, a stuck call could be attempted twelve times at the full timeout each
    before the turn gave up.
    """
    from openai import AzureOpenAI
    return AzureOpenAI(azure_endpoint=endpoint, api_key=key, api_version=version,
                       timeout=timeout, max_retries=0)


def _retriable(e: Exception) -> bool:
    status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
    if status in _RETRIABLE:
        return True
    n = type(e).__name__.lower()
    return any(k in n for k in ("ratelimit", "timeout", "connection", "internalserver"))


def _retry_after(e: Exception) -> float | None:
    """The wait Azure asked for on a 429, capped like our own backoff.

    The SDK honoured this header; now that its retries are off, this loop has to, or a
    rate-limited call comes straight back early and spends another attempt being refused.
    """
    try:
        v = getattr(getattr(e, "response", None), "headers", {}).get("retry-after")
        return min(max(float(v), 0.0), 20.0) if v is not None else None
    except Exception:
        return None


def chat(cl, **kw):
    delay = 2.0
    for attempt in range(_MAX_RETRIES):
        try:
            return cl.chat.completions.create(**windows.fill_request(kw))
        except Exception as e:  # noqa: BLE001
            if not _retriable(e) or attempt == _MAX_RETRIES - 1:
                raise
            wait = _retry_after(e)
            time.sleep(delay if wait is None else wait)
            delay = min(delay * 2.2, 20.0)


def _with_fallback(cl, role: str, **kw):
    """Run on the role's model; if that DEPLOYMENT is unavailable, fall back to the main one.

    Only `corroborate` and `gaps` run on the mini deployment, and only for independence —
    never because the task needs a weaker model. So if a given environment has no
    gpt-4o-mini deployment, the right behaviour is to lose the independence and keep the
    answer, not to fail the whole turn. Checked in this environment (both deployments
    respond), but an env var is a promise about someone else's infrastructure.
    """
    model = model_for(role)
    try:
        return chat(cl, model=model, **kw)
    except Exception as e:
        status = getattr(e, "status_code", None)
        missing = status in (404,) or "DeploymentNotFound" in str(e) or "does not exist" in str(e)
        if model == DEPLOYMENT or not missing:
            raise
        return chat(cl, model=DEPLOYMENT, **kw)


def ask_json(cl, system: str, user: str, schema_hint: str, temperature: float = 0.0,
             role: str = "plan") -> dict:
    """One structured turn. Returns {} rather than raising on a malformed reply.

    Every agent in the swarm returns JSON so the engine can act on it in code — a phase
    that has to be parsed out of prose is a phase that silently degrades.
    """
    r = _with_fallback(cl, role, temperature=temperature,
             response_format={"type": "json_object"},
             messages=[{"role": "system", "content": system + "\n\nReply with JSON only. " + schema_hint},
                       {"role": "user", "content": user}])
    try:
        return json.loads(r.choices[0].message.content or "{}")
    except Exception:
        return {}


def stream_text(cl, system: str, user: str, temperature: float = 0.0, role: str = "synthesise"):
    """Yield prose token by token — the synthesis phase is the one the user watches."""
    s = chat(cl, model=model_for(role), temperature=temperature, stream=True,
             messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    for chunk in s:
        if not getattr(chunk, "choices", None):
            continue
        d = chunk.choices[0].delta
        if d and getattr(d, "content", None):
            yield d.content
