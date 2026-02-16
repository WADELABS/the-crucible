"""
Authentication and Authorization Module for The Crucible.

This module provides user authentication, role-based access control,
and API key management.
"""

from crucible.auth.authentication import (
    UserAuthenticator,
    User,
    APIKey,
    Session,
    Role,
    Permission,
    ROLE_PERMISSIONS,
)

__all__ = [
    "UserAuthenticator",
    "User",
    "APIKey",
    "Session",
    "Role",
    "Permission",
    "ROLE_PERMISSIONS",
]
