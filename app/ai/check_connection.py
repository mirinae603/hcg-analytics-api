"""Check the AI Analyst's Azure OpenAI settings in a few seconds, without running a question.

    .venv/bin/python -m app.ai.check_connection

Reads the same settings the app reads (.env in the current folder, then the environment) and
makes one tiny call to each deployment: the main model, the mini model and the embeddings
model. Prints OK or the exact reason it failed (wrong key or endpoint, missing deployment,
unsupported API version). The key is never printed; only its length, because a key pasted
together with the text around it is the commonest mistake (an Azure key is 32 or 84
characters with no spaces).
"""
from __future__ import annotations

import os
import sys


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
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
