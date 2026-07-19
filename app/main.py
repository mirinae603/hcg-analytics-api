from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load local .env (gitignored) so AZURE_OPENAI_* etc. are available in dev.
# On Render these come from the dashboard env, so a missing .env is fine.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from app.api.api_router import api_router
from app.core.config import settings

app = FastAPI(title=settings.PROJECT_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.on_event("startup")
def _warm_caches() -> None:
    """Precompute the heavy portfolio overviews in a background thread on boot, so the
    first user request after a (free-tier) cold start is instant instead of paying the
    one-time 255k-row parquet load. Daemon thread → never blocks startup / health."""
    import threading

    def _run() -> None:
        try:
            from app.api import legacy_kpi, kpi_generic
            legacy_kpi.warmup()
            kpi_generic.warmup()
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True, name="cache-warmup").start()
