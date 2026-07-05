# user_service.py
import pyodbc
import pandas as pd
import hashlib
from dotenv import load_dotenv
import logging, os
from fastapi import HTTPException
from typing import Dict, Optional, List
from enum import Enum

load_dotenv()

class UserStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

class UserService:
    def __init__(self):
        # Azure SQL Database connection details
        self.server = 'bideasy-dashboard-server.database.windows.net,1433'
        self.database = 'bideasy-analytics-kpi'
        self.username = os.getenv("DATABASE_USERNAME")
        self.password = os.getenv("DATABASE_PASSWORD")
        self.driver = '{ODBC Driver 17 for SQL Server}'
        
        self.conn_str = (
            f'DRIVER={self.driver};'
            f'SERVER={self.server};'
            f'DATABASE={self.database};'
            f'UID={self.username};'
            f'PWD={self.password};'
            f'Encrypt=yes;'
            f'TrustServerCertificate=no;'
            f'Connection Timeout=60;'
        )
        
        # Initialize database table
        self._initialize_users_table()
    
    def _initialize_users_table(self):
        """Create users table if it doesn't exist"""
        create_table_query = """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='users' AND xtype='U')
        CREATE TABLE users (
            id INT IDENTITY(1,1) PRIMARY KEY,
            firstName NVARCHAR(100) NOT NULL,
            lastName NVARCHAR(100) NOT NULL,
            email NVARCHAR(255) UNIQUE NOT NULL,
            password_hash NVARCHAR(255) NOT NULL,
            status NVARCHAR(20) DEFAULT 'pending',
            created_at DATETIME DEFAULT GETDATE(),
            updated_at DATETIME DEFAULT GETDATE()
        )
        """
        
        try:
            with pyodbc.connect(self.conn_str) as conn:
                cursor = conn.cursor()
                cursor.execute(create_table_query)
                conn.commit()
                logging.info("Users table initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing users table: {str(e)}")
            raise
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    async def create_user(self, user_data) -> Dict:
        """Create a new user in the database"""
        # Validate input
        if not user_data.firstName.strip():
            raise HTTPException(status_code=400, detail="First name is required")
        
        if not user_data.lastName.strip():
            raise HTTPException(status_code=400, detail="Last name is required")
        
        if len(user_data.password) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters long")
        
        try:
            with pyodbc.connect(self.conn_str) as conn:
                cursor = conn.cursor()
                
                # Check if user already exists
                check_query = "SELECT COUNT(*) FROM users WHERE email = ?"
                cursor.execute(check_query, (user_data.email,))
                if cursor.fetchone()[0] > 0:
                    raise HTTPException(status_code=400, detail="Email already registered")
                
                # Insert new user
                insert_query = """
                INSERT INTO users (firstName, lastName, email, password_hash, status)
                VALUES (?, ?, ?, ?, ?)
                """
                cursor.execute(insert_query, (
                    user_data.firstName.strip(),
                    user_data.lastName.strip(),
                    user_data.email,
                    self.hash_password(user_data.password),
                    UserStatus.PENDING
                ))
                conn.commit()
                
                logging.info(f"User {user_data.email} registered successfully with pending status")
                
                return {
                    "message": "Account created successfully! Your account is under review and pending approval.",
                    "user": {
                        "email": user_data.email,
                        "firstName": user_data.firstName.strip(),
                        "lastName": user_data.lastName.strip(),
                        "status": UserStatus.PENDING
                    }
                }
        
        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Database error during user creation: {str(e)}")
            raise HTTPException(status_code=500, detail="Database error during registration")
    
    async def authenticate_user(self, credentials) -> Dict:
        """Authenticate user login"""
        # Validate input
        if not credentials.email:
            raise HTTPException(status_code=400, detail="Email is required")
        
        if not credentials.password:
            raise HTTPException(status_code=400, detail="Password is required")
        
        try:
            with pyodbc.connect(self.conn_str) as conn:
                cursor = conn.cursor()
                
                # Get user from database
                query = """
                SELECT firstName, lastName, email, password_hash, status 
                FROM users 
                WHERE email = ?
                """
                cursor.execute(query, (credentials.email,))
                user_row = cursor.fetchone()
                
                if not user_row:
                    logging.warning(f"Login attempt with non-existent email: {credentials.email}")
                    raise HTTPException(status_code=401, detail="Invalid email or password")
                
                # Convert row to dictionary
                print("Fetched User Row : ", user_row)
                user = {
                "firstName": user_row[0],
                "lastName": user_row[1],        
                "email": user_row[2],           
                "password_hash": user_row[3],   
                "status": user_row[4]          
            }
                # Verify password
                if self.hash_password(credentials.password) != user["password_hash"]:
                    logging.warning(f"Failed login attempt for email: {credentials.email}")
                    raise HTTPException(status_code=401, detail="Invalid email or password")
                
                # Check if user is approved
                if user["status"] != UserStatus.APPROVED:
                    logging.warning(f"Login attempt with pending/rejected account: {credentials.email}")
                    raise HTTPException(
                        status_code=403, 
                        detail="Your Account Approval is Pending ..."
                    )
                
                logging.info(f"User {credentials.email} logged in successfully")
                
                return {
                    "message": "Login successful!",
                    "user": {
                        "email": user["email"],
                        "firstName": user["firstName"],
                        "lastName": user["lastName"],
                        "status": user["status"]
                    }
                }
        
        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Database error during authentication: {str(e)}")
            raise HTTPException(status_code=500, detail="Database error during login")
    
    async def get_all_users(self) -> Dict:
        """Get all users with their status"""
        try:
            with pyodbc.connect(self.conn_str) as conn:
                query = """
                SELECT firstName, lastName, email, status 
                FROM users 
                ORDER BY created_at DESC
                """
                df = pd.read_sql(query, conn)
                
                safe_users = df.to_dict(orient='records')
                
                return {
                    "users": safe_users, 
                    "total": len(safe_users),
                    "pending_count": len([u for u in safe_users if u["status"] == UserStatus.PENDING]),
                    "approved_count": len([u for u in safe_users if u["status"] == UserStatus.APPROVED]),
                    "rejected_count": len([u for u in safe_users if u["status"] == UserStatus.REJECTED])
                }
        
        except Exception as e:
            logging.error(f"Database error getting all users: {str(e)}")
            raise HTTPException(status_code=500, detail="Database error")
    
    async def get_pending_users(self) -> Dict:
        """Get only pending users"""
        try:
            with pyodbc.connect(self.conn_str) as conn:
                query = """
                SELECT firstName, lastName, email, status 
                FROM users 
                WHERE status = ?
                ORDER BY created_at DESC
                """
                df = pd.read_sql(query, conn, params=[UserStatus.PENDING])
                
                pending_users = df.to_dict(orient='records')
                
                return {
                    "pending_users": pending_users, 
                    "count": len(pending_users)
                }
        
        except Exception as e:
            logging.error(f"Database error getting pending users: {str(e)}")
            raise HTTPException(status_code=500, detail="Database error")
    
    async def approve_reject_user(self, approval_request) -> Dict:
        """Approve or reject user signup"""
        email = approval_request.email
        action = approval_request.action.lower()
        
        # Validate action
        if action not in ["approve", "reject"]:
            raise HTTPException(status_code=400, detail="Action must be 'approve' or 'reject'")
        
        try:
            with pyodbc.connect(self.conn_str) as conn:
                cursor = conn.cursor()
                
                # Check if user exists and get current status
                check_query = """
                SELECT firstName, lastName, email, status 
                FROM users 
                WHERE email = ?
                """
                cursor.execute(check_query, (email,))
                user_row = cursor.fetchone()
                
                if not user_row:
                    raise HTTPException(status_code=404, detail="User not found")
                
                # In approve_reject_user method, fix this part:
                user = {
                    "firstName": user_row[0],
                    "lastName": user_row[1],    
                    "email": user_row[2],       
                    "status": user_row[3]       
                }

                
                # Check if user is currently pending
                if user["status"] != UserStatus.PENDING:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"User is already {user['status']}. Only pending users can be approved/rejected."
                    )
                
                # Update user status
                new_status = UserStatus.APPROVED if action == "approve" else UserStatus.REJECTED
                update_query = """
                UPDATE users 
                SET status = ?, updated_at = GETDATE() 
                WHERE email = ?
                """
                cursor.execute(update_query, (new_status, email))
                conn.commit()
                
                message = f"User {email} has been {'approved' if action == 'approve' else 'rejected'} successfully"
                logging.info(f"Admin action: {action} user {email}")
                
                return {
                    "message": message,
                    "user": {
                        "email": user["email"],
                        "firstName": user["firstName"],
                        "lastName": user["lastName"],
                        "status": new_status
                    }
                }
        
        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Database error during user approval: {str(e)}")
            raise HTTPException(status_code=500, detail="Database error during user approval")
    
    async def get_user_credentials(self) -> Dict:
        """Get user credentials for debugging - remove in production"""
        try:
            with pyodbc.connect(self.conn_str) as conn:
                query = """
                SELECT email, firstName, lastName, password_hash, status 
                FROM users 
                ORDER BY created_at DESC
                """
                df = pd.read_sql(query, conn)
                
                credentials = df.to_dict(orient='records')
                
                return {
                    "message": "WARNING: This endpoint should be removed in production",
                    "credentials": credentials,
                    "total": len(credentials)
                }
        
        except Exception as e:
            logging.error(f"Database error getting credentials: {str(e)}")
            raise HTTPException(status_code=500, detail="Database error")
    
    async def health_check(self) -> Dict:
        """Health check with user statistics"""
        try:
            with pyodbc.connect(self.conn_str) as conn:
                # Get count statistics
                query = """
                SELECT 
                    COUNT(*) as total_users,
                    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending_users,
                    SUM(CASE WHEN status = 'approved' THEN 1 ELSE 0 END) as approved_users,
                    SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as rejected_users
                FROM users
                """
                cursor = conn.cursor()
                cursor.execute(query)
                row = cursor.fetchone()
                
                return {
                    "status": "healthy", 
                    "total_users": row[0] or 0,
                    "pending_users": row or 0,
                    "approved_users": row or 0,
                    "rejected_users": row or 0
                }
        
        except Exception as e:
            logging.error(f"Database error during health check: {str(e)}")
            raise HTTPException(status_code=500, detail="Database error during health check")



    async def delete_user(self, email: str) -> Dict:
        """Delete a user from the database"""
        if not email:
            raise HTTPException(status_code=400, detail="Email is required")
        
        try:
            with pyodbc.connect(self.conn_str) as conn:
                cursor = conn.cursor()
                
                # Check if user exists
                check_query = """
                SELECT firstName, lastName, email, status 
                FROM users 
                WHERE email = ?
                """
                cursor.execute(check_query, (email,))
                user_row = cursor.fetchone()
                
                if not user_row:
                    raise HTTPException(status_code=404, detail="User not found")
                
                user = {
                    "firstName": user_row[0],
                    "lastName": user_row[1],
                    "email": user_row[2],
                    "status": user_row[3]
                }
                
                # Delete the user
                delete_query = "DELETE FROM users WHERE email = ?"
                cursor.execute(delete_query, (email,))
                rows_affected = cursor.rowcount
                conn.commit()
                
                if rows_affected == 0:
                    raise HTTPException(status_code=404, detail="User not found or already deleted")
                
                logging.info(f"User {email} deleted successfully")
                
                return {
                    "message": f"User {email} has been deleted successfully",
                    "deleted_user": {
                        "email": user["email"],
                        "firstName": user["firstName"],
                        "lastName": user["lastName"],
                        "status": user["status"]
                    }
                }
        
        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Database error during user deletion: {str(e)}")
            raise HTTPException(status_code=500, detail="Database error during user deletion")

    async def bulk_delete_users(self, emails: List[str]) -> Dict:
        """Delete multiple users from the database"""
        if not emails:
            raise HTTPException(status_code=400, detail="Email list cannot be empty")
        
        try:
            with pyodbc.connect(self.conn_str) as conn:
                cursor = conn.cursor()
                
                deleted_users = []
                not_found_emails = []
                
                for email in emails:
                    # Check if user exists
                    check_query = """
                    SELECT firstName, lastName, email, status 
                    FROM users 
                    WHERE email = ?
                    """
                    cursor.execute(check_query, (email,))
                    user_row = cursor.fetchone()
                    
                    if user_row:
                        user = {
                            "firstName": user_row[0],
                            "lastName": user_row[1],
                            "email": user_row[2],
                            "status": user_row[3]
                        }
                        deleted_users.append(user)
                    else:
                        not_found_emails.append(email)
                
                if deleted_users:
                    # Create placeholders for the IN clause
                    placeholders = ','.join(['?' for _ in emails])
                    delete_query = f"DELETE FROM users WHERE email IN ({placeholders})"
                    cursor.execute(delete_query, emails)
                    conn.commit()
                    
                    logging.info(f"Bulk deleted {len(deleted_users)} users")
                
                return {
                    "message": f"Bulk deletion completed. {len(deleted_users)} users deleted.",
                    "deleted_users": deleted_users,
                    "not_found_emails": not_found_emails,
                    "deleted_count": len(deleted_users),
                    "not_found_count": len(not_found_emails)
                }
        
        except Exception as e:
            logging.error(f"Database error during bulk deletion: {str(e)}")
            raise HTTPException(status_code=500, detail="Database error during bulk deletion")
