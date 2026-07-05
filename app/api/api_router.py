from fastapi import APIRouter

from app.api import kpi_generic, sales_forecast_kpi, replenishment_and_aging_risk
from app.api import dashboard_summary, legacy_kpi, authenticate

api_router = APIRouter()

# Original-frontend KPI contract (named chart + table endpoints)
api_router.include_router(legacy_kpi.router, tags=["KPI (original contract)"])
# Registry-driven KPI endpoints (used by the alt UI) + meta
api_router.include_router(kpi_generic.router, tags=["KPIs"])
# Forecasting
api_router.include_router(sales_forecast_kpi.router, tags=["Forecast KPI"])
api_router.include_router(replenishment_and_aging_risk.router, tags=["Replenishment & Aging Risk"])
# Executive summary + admin
api_router.include_router(dashboard_summary.router, tags=["Dashboard Summary"])


@api_router.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


# Auth — local JSON-backed user store (seeds an approved admin: admin@hcg.com).
# /signin, /signup, /admin/* — same contract the original frontend expects.
api_router.include_router(authenticate.router, tags=["Auth"])
