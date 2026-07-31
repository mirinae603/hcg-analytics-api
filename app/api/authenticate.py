from fastapi import HTTPException, APIRouter, Depends
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from typing import Dict, Optional, List
import logging
from enum import Enum

from app.core.database import get_db
from app.core.deps import get_current_user, require_admin
from app.models import User as UserModel
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

class SignInResponse(BaseModel):
    message: str
    token: str
    token_type: str = "bearer"
    user: Dict

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
async def signup(user_data: SignUpRequest, db: Session = Depends(get_db)):
    try:
        result = await user_service.create_user(db, user_data)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Signup error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during registration")

@router.post("/signin", response_model=SignInResponse)
async def signin(credentials: SignInRequest, db: Session = Depends(get_db)):
    try:
        result = await user_service.authenticate_user(db, credentials)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Signin error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during login")

@router.get("/me")
async def get_me(current_user: UserModel = Depends(get_current_user)):
    """Resolve the bearer token to the calling user — for the frontend to verify/refresh
    a stored token's identity without re-authenticating."""
    return {"user": user_service.safe_user(current_user)}

@router.get("/admin/users")
async def get_all_users(db: Session = Depends(get_db), _admin: UserModel = Depends(require_admin)):
    try:
        result = await user_service.get_all_users(db)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Get users error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/admin/pending-users")
async def get_pending_users(db: Session = Depends(get_db), _admin: UserModel = Depends(require_admin)):
    try:
        result = await user_service.get_pending_users(db)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Get pending users error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/admin/approve-user")
async def approve_user(approval_request: ApprovalRequest, db: Session = Depends(get_db), _admin: UserModel = Depends(require_admin)):
    try:
        result = await user_service.approve_reject_user(db, approval_request)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Approve user error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during user approval")

@router.get("/admin/credentials")
async def get_user_credentials(db: Session = Depends(get_db), _admin: UserModel = Depends(require_admin)):
    """
    Admin-only. Despite the historical name this never returns password hashes —
    only the same safe user fields as /admin/users.
    """
    try:
        result = await user_service.get_user_credentials(db)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Get credentials error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    try:
        result = await user_service.health_check(db)
        return result
    except Exception as e:
        logging.error(f"Health check error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")



@router.delete("/admin/delete-user")
async def delete_user(delete_request: DeleteUserRequest, db: Session = Depends(get_db), _admin: UserModel = Depends(require_admin)):
    """Delete a single user account"""
    try:
        result = await user_service.delete_user(db, delete_request.email)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Delete user error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during user deletion")

@router.delete("/admin/delete-user/{email}")
async def delete_user_by_path(email: str, db: Session = Depends(get_db), _admin: UserModel = Depends(require_admin)):
    """Delete a user account by email in path parameter"""
    try:
        result = await user_service.delete_user(db, email)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Delete user error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during user deletion")

@router.delete("/admin/bulk-delete-users")
async def bulk_delete_users(bulk_delete_request: BulkDeleteRequest, db: Session = Depends(get_db), _admin: UserModel = Depends(require_admin)):
    """Delete multiple user accounts"""
    try:
        result = await user_service.bulk_delete_users(db, bulk_delete_request.emails)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Bulk delete users error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during bulk deletion")
