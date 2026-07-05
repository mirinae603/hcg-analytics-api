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


settings = Settings()
