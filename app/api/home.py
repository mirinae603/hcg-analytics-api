from fastapi import APIRouter
from app.services.home_service import get_home_message

router = APIRouter()

@router.get("/")
def read_home():
    return {"message": get_home_message()}