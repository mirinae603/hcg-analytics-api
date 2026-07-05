from fastapi import HTTPException, APIRouter
from pydantic import BaseModel, EmailStr
from typing import Dict, Optional, List
import logging
from enum import Enum
from app.services.authentication_service import UserService

router = APIRouter()

class UserStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

class SignUpRequest(BaseModel):
    firstName: str
    lastName: str
    email: EmailStr
    password: str

class SignInRequest(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    message: str
    user: Optional[Dict] = None

class ApprovalRequest(BaseModel):
    email: EmailStr
    action: str  # "approve" or "reject"

class PendingUser(BaseModel):
    firstName: str
    lastName: str
    email: EmailStr
    status: UserStatus

class DeleteUserRequest(BaseModel):
    email: EmailStr

class BulkDeleteRequest(BaseModel):
    emails: List[EmailStr]

# Initialize user service
user_service = UserService()

@router.post("/signup", response_model=UserResponse)
async def signup(user_data: SignUpRequest):
    try:
        result = await user_service.create_user(user_data)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Signup error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during registration")

@router.post("/signin", response_model=UserResponse)
async def signin(credentials: SignInRequest):
    try:
        result = await user_service.authenticate_user(credentials)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Signin error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during login")

@router.get("/admin/users")
async def get_all_users():
    try:
        result = await user_service.get_all_users()
        return result
    except Exception as e:
        logging.error(f"Get users error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/admin/pending-users")
async def get_pending_users():
    try:
        result = await user_service.get_pending_users()
        return result
    except Exception as e:
        logging.error(f"Get pending users error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/admin/approve-user")
async def approve_user(approval_request: ApprovalRequest):
    try:
        result = await user_service.approve_reject_user(approval_request)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Approve user error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during user approval")

@router.get("/admin/credentials")
async def get_user_credentials():
    """
    WARNING: This endpoint exposes user credentials and should only be used for debugging.
    Remove this endpoint in production environments.
    """
    try:
        result = await user_service.get_user_credentials()
        return result
    except Exception as e:
        logging.error(f"Get credentials error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/health")
async def health_check():
    try:
        result = await user_service.health_check()
        return result
    except Exception as e:
        logging.error(f"Health check error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")



@router.delete("/admin/delete-user")
async def delete_user(delete_request: DeleteUserRequest):
    """Delete a single user account"""
    try:
        result = await user_service.delete_user(delete_request.email)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Delete user error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during user deletion")

@router.delete("/admin/delete-user/{email}")
async def delete_user_by_path(email: str):
    """Delete a user account by email in path parameter"""
    try:
        result = await user_service.delete_user(email)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Delete user error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during user deletion")

@router.delete("/admin/bulk-delete-users")
async def bulk_delete_users(bulk_delete_request: BulkDeleteRequest):
    """Delete multiple user accounts"""
    try:
        result = await user_service.bulk_delete_users(bulk_delete_request.emails)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Bulk delete users error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during bulk deletion")
