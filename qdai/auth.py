"""Local authentication for Quantum_DropOut_AI.

What this is: a small account store for a single deployment. Passwords are
never stored - only a random per-user salt and a PBKDF2-HMAC-SHA256 hash - and
comparisons are constant-time.

What this is NOT: production identity management. Accounts live in a JSON file
on the machine running the app, sessions live in Streamlit session state (a
browser refresh signs you out), and the lockout counter is per browser session,
so it slows guessing but does not stop a determined attacker. For a public
deployment use Streamlit's built-in OIDC login (st.login) with a real identity
provider instead.

This module knows nothing about the ML pipeline.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
import time
from datetime import datetime, timezone

from .config import AUTH_DIR, AUTH_FILE, LOCKOUT_SECONDS, MAX_FAILED_LOGINS, PBKDF2_ITERATIONS

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthError(ValueError):
    """A problem the user can fix; the message is safe to show as-is."""


# ------------------------------------------------------------------ storage
def _load() -> dict:
    try:
        with open(AUTH_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError):
        raise AuthError("The account store could not be read. Delete the .auth folder to reset it.")


def _save(users: dict) -> None:
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=AUTH_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(users, fh, indent=2)
        os.replace(tmp, AUTH_FILE)
        try:
            os.chmod(AUTH_FILE, 0o600)
        except OSError:
            pass
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _hash(password: str, salt: bytes, iterations: int) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations).hex()


def user_count() -> int:
    try:
        return len(_load())
    except AuthError:
        return 0


# ------------------------------------------------------------------ actions
def register(name: str, email: str, password: str, confirm: str) -> dict:
    name, email = name.strip(), email.strip().lower()
    if not name:
        raise AuthError("Enter your full name.")
    if not _EMAIL.match(email):
        raise AuthError("Enter a valid email address, for example name@school.edu.")
    if len(password) < 8:
        raise AuthError("Use a password with at least 8 characters.")
    if not (any(c.isalpha() for c in password) and any(c.isdigit() for c in password)):
        raise AuthError("Include at least one letter and one number in the password.")
    if password != confirm:
        raise AuthError("The two passwords do not match.")
    users = _load()
    if email in users:
        raise AuthError("An account with this email already exists. Sign in instead.")
    salt = secrets.token_bytes(16)
    users[email] = {
        "name": name,
        "salt": salt.hex(),
        "iterations": PBKDF2_ITERATIONS,
        "hash": _hash(password, salt, PBKDF2_ITERATIONS),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _save(users)
    return {"email": email, "name": name}


def _find(users: dict, identifier: str) -> str | None:
    """Match a full email, or the part before '@' when that is unambiguous."""
    ident = identifier.strip().lower()
    if ident in users:
        return ident
    if "@" not in ident:
        matches = [e for e in users if e.split("@")[0] == ident]
        if len(matches) == 1:
            return matches[0]
    return None


def authenticate(identifier: str, password: str) -> dict:
    users = _load()
    key = _find(users, identifier) if identifier.strip() else None
    # Always run one hash so timing does not reveal whether the account exists.
    rec = users.get(key) if key else None
    salt = bytes.fromhex(rec["salt"]) if rec else b"\x00" * 16
    iters = rec["iterations"] if rec else PBKDF2_ITERATIONS
    digest = _hash(password, salt, iters)
    if rec and hmac.compare_digest(digest, rec["hash"]):
        return {"email": key, "name": rec["name"]}
    raise AuthError("That email or password is not correct. Check both and try again.")


# ---------------------------------------------------- per-session lockout
def lockout_remaining(state) -> int:
    """Seconds left on the current lockout (0 when sign-in is allowed)."""
    until = state.get("auth_locked_until", 0.0)
    return max(0, int(round(until - time.time())))


def record_failure(state) -> None:
    state["auth_failures"] = state.get("auth_failures", 0) + 1
    if state["auth_failures"] >= MAX_FAILED_LOGINS:
        state["auth_locked_until"] = time.time() + LOCKOUT_SECONDS
        state["auth_failures"] = 0


def record_success(state) -> None:
    state["auth_failures"] = 0
    state["auth_locked_until"] = 0.0
