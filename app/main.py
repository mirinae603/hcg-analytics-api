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
