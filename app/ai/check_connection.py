"""Check the AI Analyst's Azure OpenAI settings in a few seconds, without running a question.

    .venv/bin/python -m app.ai.check_connection

Reads the same settings the app reads (.env in the current folder, then the environment) and
makes one tiny call to each deployment: the main model, the mini model, the embeddings
model and, when it is set up, the voice (speech-to-text) model, which is sent one second of
silence. Prints OK or the exact reason it failed (wrong key or endpoint, missing deployment,
unsupported API version). The key is never printed; only its length, because a key pasted
together with the text around it is the commonest mistake (an Azure key is 32 or 84
characters with no spaces).
"""
from __future__ import annotations

import io
import os
import sys
import wave


def _reason(e: Exception) -> str:
    status = getattr(e, "status_code", None)
    body = getattr(e, "body", None)
    msg = str((body.get("message") if isinstance(body, dict) else None) or e)
    hint = {401: "the key or the endpoint is wrong",
            403: "the key has no access to this resource",
            404: "no deployment with this name on this resource (or the endpoint is wrong)",
            400: "the request was refused, often an API version this resource does not support",
            429: "rate-limited; the settings themselves are fine"}.get(status, "")
    return f"{type(e).__name__} {status or ''} — {hint + ': ' if hint else ''}{msg[:200]}"


def main() -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.getcwd(), ".env"))
    except ImportError:
        pass
    from openai import AzureOpenAI

    key = os.getenv("AZURE_OPENAI_API_KEY") or ""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://ed-gpt.openai.azure.com")
    version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    main_d = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    mini_d = (os.getenv("AZURE_OPENAI_MINI_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT_MINI")
              or "gpt-4o-mini")
    embed_d = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-large")

    print(f"endpoint    {endpoint}")
    print(f"api version {version}")
    print(f"key         {'set' if key else 'MISSING'}, {len(key)} characters"
          + (" — contains spaces, so more than the key was pasted" if " " in key.strip() else ""))
    if not key:
        return 1

    client = AzureOpenAI(azure_endpoint=endpoint, api_key=key.strip(), api_version=version,
                         timeout=30, max_retries=0)
    ok = True
    for label, dep in (("main model", main_d), ("mini model", mini_d)):
        try:
            r = client.chat.completions.create(model=dep, max_tokens=3, temperature=0,
                                               messages=[{"role": "user", "content": "Reply OK"}])
            print(f"{label:<11} {dep}: OK ({r.model})")
        except Exception as e:  # noqa: BLE001 — every failure is reported, none is fatal
            ok = False
            print(f"{label:<11} {dep}: FAILED — {_reason(e)}")
    try:
        client.embeddings.create(model=embed_d, input=["ok"])
        print(f"embeddings  {embed_d}: OK")
    except Exception as e:  # noqa: BLE001
        # optional: routing falls back to keyword rules without it
        print(f"embeddings  {embed_d}: FAILED (optional, the chat still works) — {_reason(e)}")
    return 0 if (_check_voice() and ok) else 1


def _silence_wav(seconds: float = 1.0, rate: int = 16000) -> bytes:
    """A short silent mono WAV: enough for the transcribe deployment to answer, no speech."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


def _check_voice() -> bool:
    """The voice model, read the way POST /ai/transcribe reads it (app/ai/transcribe.py).
    Not set up is not a failure: the chat just shows no mic."""
    from openai import AzureOpenAI
    from app.ai import transcribe
    cfg = transcribe.settings()
    if cfg is None:
        print("voice       not set up (AZURE_OPENAI_TRANSCRIBE_DEPLOYMENT is empty): the chat shows no mic")
        return True
    if cfg.endpoint != os.getenv("AZURE_OPENAI_ENDPOINT", cfg.endpoint):
        print(f"voice       on its own resource {cfg.endpoint}")
    client = AzureOpenAI(azure_endpoint=cfg.endpoint, api_key=cfg.key, api_version=cfg.api_version,
                         timeout=30, max_retries=0)
    try:
        r = client.audio.transcriptions.create(model=cfg.deployment, file=("check.wav", _silence_wav(), "audio/wav"),
                                               response_format="json")
        heard = (getattr(r, "text", "") or "").strip()
        print(f"voice       {cfg.deployment}: OK (api version {cfg.api_version}"
              + (f"; heard {heard!r} in silence" if heard else "") + ")")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"voice       {cfg.deployment}: FAILED — {_reason(e)}")
        return False


if __name__ == "__main__":
    sys.exit(main())
