import hashlib
import os
from typing import Optional, Dict

SALT = "smart_ai_evaluator_salt_2026"

def hash_password(password: str) -> str:
    """Hashes password using SHA-256 with a salt."""
    salted = (SALT + password).encode('utf-8')
    return hashlib.sha256(salted).hexdigest()

def verify_password(password: str, password_hash: str) -> bool:
    """Verifies a plain password against a stored SHA-256 hash."""
    return hash_password(password) == password_hash

DEMO_USERS = {
    "teacher@smartai.edu": {
        "username": "teacher@smartai.edu",
        "name": "Prof. Sarah Jenkins",
        "password_hash": hash_password("teacher123"),
        "role": "Teacher"
    },
    "admin@smartai.edu": {
        "username": "admin@smartai.edu",
        "name": "System Administrator",
        "password_hash": hash_password("admin123"),
        "role": "Admin"
    }
}

def authenticate_user(username_or_email: str, password_input: str) -> Optional[Dict[str, str]]:
    """Validates user credentials against demo users or database."""
    user = DEMO_USERS.get(username_or_email.strip().lower())
    if user:
        if verify_password(password_input, user["password_hash"]):
            return {
                "username": user["username"],
                "name": user["name"],
                "role": user["role"]
            }
    return None
