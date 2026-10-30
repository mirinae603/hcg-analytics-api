import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "HCG Supply Chain Analytics Backend"

    # --- Data layout ---------------------------------------------------------
    # Root of HCG's raw SAP extracts for the curated import (python -m app.etl.run_etl): the
    # repo's master/ folder, the same root as FORMULARY_RAW_DIR below. app/etl/ingest.py finds
    # the Consumption, Inventory, GRN and PO folders anywhere under it (HCG's files sit one
    # hashed folder down). Override via env RAW_DATA_DIR.
    RAW_DATA_DIR: str = str(Path(__file__).resolve().parents[2] / "master")
    # Curated parquet (fact + dim tables) produced by the ETL.
    CURATED_DIR: str = str(Path(__file__).resolve().parents[2] / "app" / "data" / "curated")
    # Pre-computed per-KPI aggregate parquet tables.
    KPI_DIR: str = str(Path(__file__).resolve().parents[2] / "app" / "data" / "kpi")

    # --- Formulary compliance (python -m app.etl.formulary) ------------------
    # HCG's raw monthly pharmacy sales + item/payer masters. Line-level patient data:
    # git- and docker-ignored, read only by the build run by hand.
    FORMULARY_RAW_DIR: str = str(Path(__file__).resolve().parents[2] / "master")
    # Line-level parquet the build keeps between runs — outside the repo on purpose.
    FORMULARY_CACHE_DIR: str = os.path.expanduser("~/.cache/hcg-analytics/formulary")
    # Aggregated tables the API serves (committed). Not app/data/kpi: the AI warehouse
    # auto-loads every parquet there and the ontology staleness test would fail.
    FORMULARY_DATA_DIR: str = str(Path(__file__).resolve().parents[2] / "app" / "data" / "formulary")
    # Hand-maintained lookups (hospital_map.csv, payer_overrides.csv), committed.
    REFERENCE_DIR: str = str(Path(__file__).resolve().parents[2] / "app" / "data" / "reference")

    # --- Spend analysis (python -m app.etl.spend) -------------------------------
    # Aggregated scheme-brand tables the API serves (committed), built from the formulary
    # line cache above and HCG's Spend Analysis Main Sheet. Not app/data/kpi, same reason.
    SPEND_DATA_DIR: str = str(Path(__file__).resolve().parents[2] / "app" / "data" / "spend")

    # --- Purchase analysis (python -m app.etl.purchase) -------------------------
    # Aggregated tables built from HCG's GRN (goods receipt) extracts and out-of-formulary
    # workbooks under FORMULARY_RAW_DIR (committed). Not app/data/kpi, same reason.
    PURCHASE_DATA_DIR: str = str(Path(__file__).resolve().parents[2] / "app" / "data" / "purchase")
    # One parquet per GRN workbook (line level, no GR/PO/invoice numbers), keyed on the file's
    # path, size and mtime so each workbook is read once — outside the repo on purpose.
    PURCHASE_CACHE_DIR: str = os.path.expanduser("~/.cache/hcg-analytics/purchase")

    # --- CORS ------------------------------------------------------------------
    # Comma-separated exact origins, plus a regex for the Static Web Apps hosts (the
    # default *.azurestaticapps.net host and its per-PR preview environments).
    CORS_ORIGINS: str = "http://localhost:3000,https://hcg.bidezy.com"
    CORS_ORIGIN_REGEX: str = r"https://.*\.azurestaticapps\.net"

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
    # Password given to admin@hcg.com when startup seeds it (only if it does not exist).
    # DEV-ONLY DEFAULT — every data route needs a login, so production MUST set this.
    SEED_ADMIN_PASSWORD: str = "admin123"


settings = Settings()
