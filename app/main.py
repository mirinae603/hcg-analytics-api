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

# NOTE: NO startup cache-warming. On the 512 MB free tier, eagerly loading fact_grn +
# fact_po (168 MB) plus every summary at boot spikes RSS past the limit → OOM crash-loop.
# Tables load lazily per request; the per-endpoint result caches keep warm navigation
# instant, and the big fact tables are not held resident (see data_access.load).

# After each request, hand the memory freed by transient big-table loads back to the OS.
# glibc keeps freed heap by default (RSS stays high on a 512 MB box); malloc_trim(0)
# releases it. Linux-only — resolves to a no-op everywhere else.
import ctypes
import ctypes.util as _cu

try:
    _libc = ctypes.CDLL(_cu.find_library("c"))
    _HAS_TRIM = hasattr(_libc, "malloc_trim")
except Exception:
    _libc, _HAS_TRIM = None, False


@app.middleware("http")
async def _reclaim_memory(request, call_next):
    response = await call_next(request)
    if _HAS_TRIM:
        try:
            _libc.malloc_trim(0)
        except Exception:
            pass
    return response
