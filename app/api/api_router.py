from fastapi import APIRouter, Depends

from app.api import kpi_generic, sales_forecast_kpi, replenishment_and_aging_risk
from app.api import dashboard_summary, legacy_kpi, authenticate, chat, formulary, spend, brand_rules, consumables
from app.ai import router as ai_router
from app.core.deps import get_current_user

api_router = APIRouter()

# Every data route needs a signed-in, approved user (Authorization: Bearer <token>).
# Left open: /health and the auth router below (signin/signup; its /me and /admin/*
# routes check the token themselves). The chat router also checks per route.
_login = [Depends(get_current_user)]

# Original-frontend KPI contract (named chart + table endpoints)
api_router.include_router(legacy_kpi.router, dependencies=_login, tags=["KPI (original contract)"])
# Registry-driven KPI endpoints (used by the alt UI) + meta
api_router.include_router(kpi_generic.router, dependencies=_login, tags=["KPIs"])
# Forecasting
api_router.include_router(sales_forecast_kpi.router, dependencies=_login, tags=["Forecast KPI"])
api_router.include_router(replenishment_and_aging_risk.router, dependencies=_login, tags=["Replenishment & Aging Risk"])
# Executive summary + admin (POST /admin/refresh-data also needs x-admin-token)
api_router.include_router(dashboard_summary.router, dependencies=_login, tags=["Dashboard Summary"])
# Formulary compliance (HIS pharmacy sales; GET /formulary/validation is admin only)
api_router.include_router(formulary.router, dependencies=_login, tags=["Formulary Compliance"])
# HCG's brand rules per payer (Task 4): /formulary/rules/*
api_router.include_router(brand_rules.router, dependencies=_login, tags=["Brand Rules"])
# Spend analysis: Attis brands for scheme patients (GET /spend/validation is admin only)
api_router.include_router(spend.router, dependencies=_login, tags=["Spend Analysis"])
# Consumables issued to each department (SAP cost centre), on the Consumption page
api_router.include_router(consumables.router, dependencies=_login, tags=["Consumables by Department"])


@api_router.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


# Auth — local JSON-backed user store (seeds an approved admin: admin@hcg.com).
# /signin, /signup, /admin/* — same contract the original frontend expects.
api_router.include_router(authenticate.router, tags=["Auth"])

# AI Analyst — Azure OpenAI tool-calling over the deterministic analytics catalog (SSE).
api_router.include_router(ai_router.router, dependencies=_login, tags=["AI Analyst"])

# AI Analyst chat persistence — sessions/messages, common/shared across all logged-in users.
api_router.include_router(chat.router, tags=["AI Chat Sessions"])
