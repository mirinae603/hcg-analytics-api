# app/core/security.py — password hashing.
#
# bcrypt via passlib replaces the old unsalted-SHA-256 scheme in
# authentication_service.py. A legacy-verify fallback is kept for defensiveness: IF a
# migrated user record ever turns up with an old sha256 password_hash (none does today
# — the only pre-existing record was the seed admin, which is re-seeded fresh with a
# bcrypt hash during migration, see app/core/seed.py), signing in with the correct
# plaintext still succeeds and the stored hash is transparently upgraded to bcrypt on
# that successful login, rather than locking the user out.
from __future__ import annotations

import hashlib

from passlib.context import CryptContext

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """True if `password` matches the bcrypt `password_hash`. Never raises."""
    try:
        return _pwd_context.verify(password, password_hash)
    except Exception:
        return False


def is_bcrypt_hash(password_hash: str) -> bool:
    return bool(password_hash) and password_hash.startswith(("$2a$", "$2b$", "$2y$"))


def legacy_sha256(password: str) -> str:
    """The OLD unsalted hash scheme this replaces — verify-only, never used to hash anew."""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password_with_legacy_fallback(password: str, password_hash: str) -> bool:
    """Verify against bcrypt first; if the stored hash isn't bcrypt, fall back to the
    legacy sha256 comparison so pre-existing accounts aren't locked out. Callers should
    re-hash (to bcrypt) and persist on a successful legacy match."""
    if is_bcrypt_hash(password_hash):
        return verify_password(password, password_hash)
    return legacy_sha256(password) == password_hash
