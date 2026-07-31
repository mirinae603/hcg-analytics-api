from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "HCG Supply Chain Analytics Backend"

    # --- Data layout ---------------------------------------------------------
    # Root of the raw HCG SAP excel extracts (Bidezy-2). Override via env RAW_DATA_DIR.
    RAW_DATA_DIR: str = str(
        Path(__file__).resolve().parents[3] / "Bidezy-2"
    )
    # Curated parquet (fact + dim tables) produced by the ETL.
    CURATED_DIR: str = str(Path(__file__).resolve().parents[2] / "app" / "data" / "curated")
    # Pre-computed per-KPI aggregate parquet tables.
    KPI_DIR: str = str(Path(__file__).resolve().parents[2] / "app" / "data" / "kpi")

    # --- Forecasting ---------------------------------------------------------
    FORECAST_HORIZON_MONTHS: int = 3
    FORECAST_Z: float = 1.645  # ~95% one-sided band

    # --- Admin ---------------------------------------------------------------
    ADMIN_REFRESH_TOKEN: str = "change-me"

    # --- Database (SQLite via SQLAlchemy — users, chat sessions/messages) ----
    # A local file next to the old users.json, same gitignored-local-store convention.
    # Override via env DATABASE_URL (e.g. to point at Postgres later) if ever needed.
    DATABASE_URL: str = "sqlite:///" + str(
        Path(__file__).resolve().parents[1] / "data" / "app.db"
    )

    # --- Auth (JWT) ------------------------------------------------------------
    # DEV-ONLY DEFAULT — a real deployment MUST set JWT_SECRET_KEY via env to a long
    # random secret. Left with a fallback so local dev works out of the box; the
    # fallback must never be relied on in production.
    JWT_SECRET_KEY: str = "dev-only-insecure-default-secret-CHANGE-ME-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 720  # 12 hours


settings = Settings()
