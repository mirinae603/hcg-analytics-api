# app/services/authentication_service.py — user accounts, now SQLite + bcrypt + JWT.
#
# Public method shapes are kept close to the previous JSON-backed version (same
# message/user response conventions) so the existing frontend auth pages keep working;
# the notable additions are `role` on every user payload and a real `token` returned by
# sign-in. Every method takes a `db: Session` (request-scoped, injected by FastAPI via
# Depends(get_db) in app/api/authenticate.py) instead of reading/writing a JSON file.
from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.jwt_auth import create_access_token
from app.core.security import hash_password, verify_password_with_legacy_fallback, is_bcrypt_hash
from app.models import User, UserRole, UserStatus


class UserService:
    # ---------- helpers ----------
    def _find(self, db: Session, email: str) -> Optional[User]:
        return db.query(User).filter(func.lower(User.email) == str(email).strip().lower()).first()

    def safe_user(self, u: User) -> Dict:
        return {
            "id": u.id,
            "firstName": u.first_name,
            "lastName": u.last_name,
            "email": u.email,
            "role": u.role,
            "status": u.status,
        }

    # ---------- public API ----------
    async def create_user(self, db: Session, user_data) -> Dict:
        if not user_data.firstName.strip():
            raise HTTPException(status_code=400, detail="First name is required")
        if not user_data.lastName.strip():
            raise HTTPException(status_code=400, detail="Last name is required")
        if len(user_data.password) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters long")

        if self._find(db, user_data.email):
            raise HTTPException(status_code=409, detail="An account with this email already exists")

        user = User(
            first_name=user_data.firstName.strip(),
            last_name=user_data.lastName.strip(),
            email=str(user_data.email).strip(),
            password_hash=hash_password(user_data.password),
            role=UserRole.MEMBER,
            status=UserStatus.PENDING,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        return {
            "message": "Registration successful! Your account is pending admin approval.",
            "user": self.safe_user(user),
        }

    async def authenticate_user(self, db: Session, credentials) -> Dict:
        if not credentials.email:
            raise HTTPException(status_code=400, detail="Email is required")
        if not credentials.password:
            raise HTTPException(status_code=400, detail="Password is required")

        user = self._find(db, credentials.email)
        if not user or not verify_password_with_legacy_fallback(credentials.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid email or password")

        # Transparent upgrade: a legacy (pre-bcrypt) hash that just verified correctly
        # is re-hashed with bcrypt so this is a one-time fallback, not a standing gap.
        if not is_bcrypt_hash(user.password_hash):
            user.password_hash = hash_password(credentials.password)
            db.commit()

        if user.status == UserStatus.PENDING:
            raise HTTPException(status_code=403, detail="Your account approval is pending. Please wait for an admin to approve it.")
        if user.status == UserStatus.REJECTED:
            raise HTTPException(status_code=403, detail="Your account request was rejected. Contact an administrator.")
        if user.status != UserStatus.APPROVED:
            raise HTTPException(status_code=403, detail="Your account is not approved.")

        token = create_access_token(user)
        return {
            "message": "Login successful!",
            "token": token,
            "token_type": "bearer",
            "user": self.safe_user(user),
        }

    async def get_all_users(self, db: Session) -> Dict:
        users = db.query(User).order_by(User.created_at.asc()).all()
        return {"message": "Users fetched successfully", "users": [self.safe_user(u) for u in users]}

    async def get_pending_users(self, db: Session) -> Dict:
        users = db.query(User).filter(User.status == UserStatus.PENDING).order_by(User.created_at.asc()).all()
        return {"users": [self.safe_user(u) for u in users]}

    async def approve_reject_user(self, db: Session, approval_request) -> Dict:
        action = str(getattr(approval_request, "action", "")).lower()
        new_status = UserStatus.APPROVED if action == "approve" else UserStatus.REJECTED

        user = self._find(db, approval_request.email)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        user.status = new_status
        db.commit()
        db.refresh(user)

        return {"message": f"User {new_status} successfully", "user": self.safe_user(user)}

    async def get_user_credentials(self, db: Session) -> Dict:
        """WARNING: debugging-only endpoint (admin-protected) — never exposes password_hash."""
        users = db.query(User).order_by(User.created_at.asc()).all()
        return {"users": [self.safe_user(u) for u in users]}

    async def health_check(self, db: Session) -> Dict:
        return {"status": "ok", "store": "sqlite", "users": db.query(User).count()}

    async def delete_user(self, db: Session, email: str) -> Dict:
        user = self._find(db, email)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        db.delete(user)
        db.commit()
        return {"message": "User deleted successfully"}

    async def bulk_delete_users(self, db: Session, emails: List[str]) -> Dict:
        lows = {str(e).strip().lower() for e in emails}
        users = db.query(User).all()
        deleted = 0
        for u in users:
            if u.email.lower() in lows:
                db.delete(u)
                deleted += 1
        db.commit()
        return {"message": f"Deleted {deleted} user(s)"}
