from fastapi import APIRouter, Query
from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd

df_aging_card = pd.read_csv("app/data/card_data/stockAgingCard.csv")
df_quantity_card = pd.read_csv("app/data/card_data/stockCard.csv")
df_value_card = pd.read_csv("app/data/card_data/stockValueCard.csv")
df_revenue_card = pd.read_csv("app/data/card_data/stockRevenueCard.csv")
df_returnrate_card = pd.read_csv("app/data/card_data/stockReturnRateCard.csv")
df_itr_card = pd.read_csv("app/data/card_data/stockITRCard.csv")
df_dih_card = pd.read_csv("app/data/card_data/stockCardDIH.csv")

router = APIRouter()

# Your existing data models (unchanged, for reference and validation)
class StockAgingData(BaseModel):
    fresh: int
    aging: int
    problem: int
    deadStock: int
    lastUpdated: datetime

class KpiStockLevelData(BaseModel):
    currentStock: int
    stockValue: float
    lastMonthRevenue: float
    maxStockValue: float
    monthlyRevenueTarget: float
    margin: float
    unit: str
    currency: str
    label: str
    lowStockThreshold: int
    location: str
    supplier: str
    lastUpdated: datetime

class ReturnRateData(BaseModel):
    currentReturnRate: float
    historicalData: Dict[str, float]
    trend: Dict[str, Any]
    targetReturnRate: float
    industryAverage: float

class DaysOnHandData(BaseModel):
    daysOnHand: int
    trend: Dict[str, Any]
    criticalThreshold: int
    optimalRange: Dict[str, int]
    category: str
    location: str
    lastCalculated: datetime

class InventoryTurnoverData(BaseModel):
    currentITR: float
    label: str
    trend: Dict[str, Any]
    targetITR: float
    industryAverage: float

class DashboardData(BaseModel):
    stockAging: StockAgingData
    kpiStockLevel: KpiStockLevelData
    returnRate: ReturnRateData
    daysOnHand: DaysOnHandData
    inventoryTurnover: InventoryTurnoverData

def fetch_aging_records(region: str) -> Dict[str, Any]:
    filtered_data = df_aging_card[df_aging_card['Plant']==region]
    return {
            "fresh": filtered_data[filtered_data['Aging Category']=='<3 Months']['Count'].values.tolist()[0],
            "aging": filtered_data[filtered_data['Aging Category']=='3+ Months']['Count'].values.tolist()[0],
            "problem": filtered_data[filtered_data['Aging Category']=='6+ Months']['Count'].values.tolist()[0],
            "deadStock": filtered_data[filtered_data['Aging Category']=='1+ Year']['Count'].values.tolist()[0],
            "lastUpdated": datetime.utcnow()
        }

def fetch_stockCard_records(region: str) -> Dict[str, Any]:
    filtered_quantity_data = df_quantity_card[df_quantity_card['Location']==region]
    filtered_value_data = df_value_card[df_value_card['Location']==region]
    filtered_revenue_data = df_revenue_card[df_revenue_card['Location']==region]
    return {
            "currentStock": filtered_quantity_data['2024-2025'].values.tolist()[0],
            "stockValue": filtered_value_data['2024-2025'].values.tolist()[0],
            "lastMonthRevenue": filtered_revenue_data['March 2025 Revenue'].values.tolist()[0],
            "maxStockValue": filtered_value_data['2023-2024'].values.tolist()[0],
            "monthlyRevenueTarget": filtered_revenue_data['March 2024 Revenue'].values.tolist()[0],
            "margin": 18.5,
            "unit": "Units",
            "currency": "₹",
            "label": "Product A Inventory",
            "lowStockThreshold":filtered_quantity_data['2023-2024'].values.tolist()[0],
            "location": region,
            "supplier": "BidEasy",
            "lastUpdated": datetime.utcnow()
        }

def fetch_returnrate_records(region: str) -> Dict[str, Any]:
    filtered_returnrate_data = df_returnrate_card[df_returnrate_card['Plant']==region]
    current = filtered_returnrate_data[(filtered_returnrate_data['Year']==2025)&(filtered_returnrate_data['Month']=="March")]['Return Rate (%)'].values.tolist()[0]
    past = filtered_returnrate_data[(filtered_returnrate_data['Year']==2025)&(filtered_returnrate_data['Month']=="February")]['Return Rate (%)'].values.tolist()[0]
    change = round(((current-past)/past)*100,2)
    status = "up" if change>=0 else "down"
    return {
            "currentReturnRate": current,
            "historicalData": {
                "thirtyDaysAgo": filtered_returnrate_data[(filtered_returnrate_data['Year']==2025)&(filtered_returnrate_data['Month']=="February")]['Return Rate (%)'].values.tolist()[0],
                "sixtyDaysAgo": filtered_returnrate_data[(filtered_returnrate_data['Year']==2025)&(filtered_returnrate_data['Month']=="January")]['Return Rate (%)'].values.tolist()[0],
                "ninetyDaysAgo": filtered_returnrate_data[(filtered_returnrate_data['Year']==2024)&(filtered_returnrate_data['Month']=="December")]['Return Rate (%)'].values.tolist()[0]
            },
            "trend": {
                "direction": status,
                "percentage": abs(change),
                "period": "30d"
            },
            "targetReturnRate": round(filtered_returnrate_data['Return Rate (%)'].min().tolist(), 2),
            "industryAverage": round(filtered_returnrate_data['Return Rate (%)'].iloc[:-1].mean().tolist(),2),
        }

def fetch_itr_records(region: str) -> Dict[str, Any]:
    filtered_itr_data = df_itr_card[df_itr_card['Plant']==region]
    current = filtered_itr_data[(filtered_itr_data['Year']==2025)&(filtered_itr_data['Month']=="March")]['ITR'].values.tolist()[0]
    past = filtered_itr_data[(filtered_itr_data['Year']==2025)&(filtered_itr_data['Month']=="February")]['ITR'].values.tolist()[0]
    change = round(((current-past)/past)*100,2)
    status = "up" if change>=0 else "down"
    return {
            "currentITR": current,
           "label": "Inventory TurnOver Ratio",
            "trend": {
                "direction": status,
                "percentage": abs(change),
                "period": "30d"
            },
            "targetITR": round(filtered_itr_data['ITR'].min().tolist(), 2),
            "industryAverage": round(filtered_itr_data['ITR'].iloc[:-1].mean().tolist(),2),
        }


def fetch_dih_records(region: str) -> Dict[str, Any]:
    filtered_dih_data = df_dih_card[df_dih_card['Location']==region]
    current = filtered_dih_data['Current DIH'].values.tolist()[0]
    past = filtered_dih_data['Last Month DIH'].values.tolist()[0]
    change = round(((current-past)/past)*100,2)
    status = "up" if change>=0 else "down"
    return {
            "daysOnHand": current,
            "trend": {
                "direction": status,
                "percentage": change,
                "period": "vs last month",
                "previousValue": past
            },
            "criticalThreshold": 7,
            "optimalRange": {"min": 60, "max": 90},
            "category": "Raw Materials",
            "location": region,
            "lastCalculated": datetime.utcnow()
        }


@router.get("/api/dashboard/all", response_model=DashboardData)
async def get_all_dashboard_data(region: Optional[str] = Query("Vijayawada")):
    return {
        "stockAging": fetch_aging_records(region),
        "kpiStockLevel": fetch_stockCard_records(region),
        "returnRate": fetch_returnrate_records(region),
        "daysOnHand": fetch_dih_records(region),
        "inventoryTurnover": fetch_itr_records(region)
    }
