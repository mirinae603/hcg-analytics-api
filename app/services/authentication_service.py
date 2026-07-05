# user_service.py — local JSON-backed user store.
# The original implementation used Azure SQL (pyodbc); that dependency was removed
# for the free Vercel/Render deploy. This keeps the SAME public interface and return
# shapes so app/api/authenticate.py works unchanged, but persists users to a JSON
# file and seeds an approved admin so sign-in works out of the box.
import json
import os
import hashlib
import logging
import threading
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List

from fastapi import HTTPException

USERS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "users.json")
_LOCK = threading.Lock()

# Seed admin so the app is usable immediately (change via signup/admin later).
SEED_ADMIN = {"firstName": "HCG", "lastName": "Admin", "email": "admin@hcg.com",
              "password": "admin123", "status": "approved"}


class UserStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class UserService:
    def __init__(self):
        os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
        self._ensure_seed()

    # ---------- storage helpers ----------
    def _load(self) -> List[Dict]:
        if not os.path.exists(USERS_FILE):
            return []
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"users.json read error: {e}")
            return []

    def _save(self, users: List[Dict]) -> None:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2)

    def _ensure_seed(self) -> None:
        with _LOCK:
            users = self._load()
            if not any(u["email"].lower() == SEED_ADMIN["email"].lower() for u in users):
                now = datetime.now(timezone.utc).isoformat()
                users.append({"firstName": SEED_ADMIN["firstName"], "lastName": SEED_ADMIN["lastName"],
                              "email": SEED_ADMIN["email"], "password_hash": self.hash_password(SEED_ADMIN["password"]),
                              "status": SEED_ADMIN["status"], "created_at": now, "updated_at": now})
                self._save(users)
                logging.info("Seeded default admin user")

    def hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    def _find(self, users: List[Dict], email: str):
        for u in users:
            if u["email"].lower() == str(email).lower():
                return u
        return None

    def _safe(self, u: Dict) -> Dict:
        return {"firstName": u.get("firstName"), "lastName": u.get("lastName"),
                "email": u.get("email"), "status": u.get("status")}

    # ---------- public API (same shapes as the original service) ----------
    async def create_user(self, user_data) -> Dict:
        if not user_data.firstName.strip():
            raise HTTPException(status_code=400, detail="First name is required")
        if not user_data.lastName.strip():
            raise HTTPException(status_code=400, detail="Last name is required")
        if len(user_data.password) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters long")
        with _LOCK:
            users = self._load()
            if self._find(users, user_data.email):
                raise HTTPException(status_code=409, detail="An account with this email already exists")
            now = datetime.now(timezone.utc).isoformat()
            user = {"firstName": user_data.firstName.strip(), "lastName": user_data.lastName.strip(),
                    "email": user_data.email, "password_hash": self.hash_password(user_data.password),
                    "status": UserStatus.PENDING.value, "created_at": now, "updated_at": now}
            users.append(user)
            self._save(users)
        return {"message": "Registration successful! Your account is pending admin approval.",
                "user": self._safe(user)}

    async def authenticate_user(self, credentials) -> Dict:
        if not credentials.email:
            raise HTTPException(status_code=400, detail="Email is required")
        if not credentials.password:
            raise HTTPException(status_code=400, detail="Password is required")
        users = self._load()
        user = self._find(users, credentials.email)
        if not user or self.hash_password(credentials.password) != user["password_hash"]:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        if user["status"] != UserStatus.APPROVED.value:
            raise HTTPException(status_code=403, detail="Your Account Approval is Pending ...")
        return {"message": "Login successful!",
                "user": {"email": user["email"], "firstName": user["firstName"],
                         "lastName": user["lastName"], "status": user["status"]}}

    async def get_all_users(self) -> Dict:
        users = self._load()
        return {"message": "Users fetched successfully", "users": [self._safe(u) for u in users]}

    async def get_pending_users(self) -> Dict:
        users = self._load()
        return {"users": [self._safe(u) for u in users if u.get("status") == UserStatus.PENDING.value]}

    async def approve_reject_user(self, approval_request) -> Dict:
        action = str(getattr(approval_request, "action", "")).lower()
        new_status = UserStatus.APPROVED.value if action == "approve" else UserStatus.REJECTED.value
        with _LOCK:
            users = self._load()
            user = self._find(users, approval_request.email)
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            user["status"] = new_status
            user["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._save(users)
        return {"message": f"User {new_status} successfully", "user": self._safe(user)}

    async def get_user_credentials(self) -> Dict:
        users = self._load()
        return {"users": [self._safe(u) for u in users]}

    async def health_check(self) -> Dict:
        return {"status": "ok", "store": "json", "users": len(self._load())}

    async def delete_user(self, email: str) -> Dict:
        with _LOCK:
            users = self._load()
            if not self._find(users, email):
                raise HTTPException(status_code=404, detail="User not found")
            users = [u for u in users if u["email"].lower() != str(email).lower()]
            self._save(users)
        return {"message": "User deleted successfully"}

    async def bulk_delete_users(self, emails: List[str]) -> Dict:
        lows = {str(e).lower() for e in emails}
        with _LOCK:
            users = self._load()
            users = [u for u in users if u["email"].lower() not in lows]
            self._save(users)
        return {"message": f"Deleted {len(lows)} user(s)"}
