"""
Authentication and Authorization System for The Crucible.

This module provides:
- User authentication with multiple backends
- Role-based access control (RBAC)
- API key generation and management
- Session management
"""

import bcrypt
import json
import logging
import secrets
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, asdict


logger = logging.getLogger(__name__)


class Role(Enum):
    """User roles with different permission levels."""
    VIEWER = "viewer"  # Can view reports only
    ANALYST = "analyst"  # Can run tests
    ADMIN = "admin"  # Full access


class Permission(Enum):
    """System permissions."""
    VIEW_REPORTS = "view_reports"
    RUN_TESTS = "run_tests"
    MODIFY_MODELS = "modify_models"
    EXPORT_DATA = "export_data"
    MANAGE_USERS = "manage_users"
    GRANT_CONSENT = "grant_consent"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.VIEWER: {
        Permission.VIEW_REPORTS,
    },
    Role.ANALYST: {
        Permission.VIEW_REPORTS,
        Permission.RUN_TESTS,
        Permission.EXPORT_DATA,
    },
    Role.ADMIN: {
        Permission.VIEW_REPORTS,
        Permission.RUN_TESTS,
        Permission.MODIFY_MODELS,
        Permission.EXPORT_DATA,
        Permission.MANAGE_USERS,
        Permission.GRANT_CONSENT,
    },
}


@dataclass
class User:
    """User account representation."""
    username: str
    email: str
    role: Role
    password_hash: str
    created_at: datetime
    last_login: Optional[datetime] = None
    active: bool = True
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        return permission in ROLE_PERMISSIONS.get(self.role, set())
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "active": self.active,
        }
        if include_sensitive:
            data["password_hash"] = self.password_hash
        return data


@dataclass
class APIKey:
    """API key for programmatic access."""
    key: str
    username: str
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime] = None
    active: bool = True
    
    def is_valid(self) -> bool:
        """Check if API key is valid."""
        if not self.active:
            return False
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return True
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "username": self.username,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "active": self.active,
        }
        if include_sensitive:
            data["key"] = self.key
        return data


@dataclass
class Session:
    """User session."""
    session_id: str
    username: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    
    def is_valid(self) -> bool:
        """Check if session is valid."""
        return datetime.now() < self.expires_at


class UserAuthenticator:
    """
    Handle user authentication for Crucible operations.
    
    Supports file-based authentication with future extensibility for
    database and OAuth backends.
    """
    
    def __init__(self, auth_backend: str = "file", storage_path: Path = Path("./auth_data")):
        """
        Initialize authenticator.
        
        Args:
            auth_backend: Authentication backend ("file", "database", "oauth")
            storage_path: Path to store authentication data (for file backend)
        """
        self.auth_backend = auth_backend
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self.sessions: Dict[str, Session] = {}
        
        if auth_backend == "file":
            self._load_users()
            self._load_api_keys()
        else:
            logger.warning(f"Backend '{auth_backend}' not fully implemented, using in-memory storage")
        
        logger.info(f"UserAuthenticator initialized with {auth_backend} backend")
    
    def _hash_password(self, password: str) -> str:
        """
        Hash password using bcrypt with salt.
        
        Args:
            password: Plain text password
            
        Returns:
            Bcrypt hash string (includes salt)
        """
        # Generate salt and hash password
        # Using default work factor of 12 rounds (2^12 iterations)
        salt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
        # Return as string for JSON serialization
        return password_hash.decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """
        Verify password against bcrypt hash.
        
        Args:
            password: Plain text password to verify
            password_hash: Stored bcrypt hash
            
        Returns:
            True if password matches hash
        """
        try:
            return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
        except (ValueError, TypeError) as e:
            logger.warning(f"Password verification failed: {e}")
            return False
    
    def _load_users(self) -> None:
        """Load users from file storage."""
        users_file = self.storage_path / "users.json"
        if not users_file.exists():
            # Create default admin user
            self.create_user("admin", "admin@crucible.local", "admin", Role.ADMIN)
            return
        
        try:
            with open(users_file, 'r') as f:
                data = json.load(f)
                for username, user_data in data.items():
                    self.users[username] = User(
                        username=user_data["username"],
                        email=user_data["email"],
                        role=Role(user_data["role"]),
                        password_hash=user_data["password_hash"],
                        created_at=datetime.fromisoformat(user_data["created_at"]),
                        last_login=datetime.fromisoformat(user_data["last_login"]) if user_data.get("last_login") else None,
                        active=user_data.get("active", True),
                    )
            logger.info(f"Loaded {len(self.users)} users")
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error loading users: {e}")
    
    def _save_users(self) -> None:
        """Save users to file storage."""
        users_file = self.storage_path / "users.json"
        try:
            data = {
                username: user.to_dict(include_sensitive=True)
                for username, user in self.users.items()
            }
            with open(users_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug("Users saved")
        except IOError as e:
            logger.error(f"Error saving users: {e}")
            raise
    
    def _load_api_keys(self) -> None:
        """Load API keys from file storage."""
        keys_file = self.storage_path / "api_keys.json"
        if not keys_file.exists():
            return
        
        try:
            with open(keys_file, 'r') as f:
                data = json.load(f)
                for key_str, key_data in data.items():
                    self.api_keys[key_str] = APIKey(
                        key=key_data["key"],
                        username=key_data["username"],
                        created_at=datetime.fromisoformat(key_data["created_at"]),
                        expires_at=datetime.fromisoformat(key_data["expires_at"]) if key_data.get("expires_at") else None,
                        last_used=datetime.fromisoformat(key_data["last_used"]) if key_data.get("last_used") else None,
                        active=key_data.get("active", True),
                    )
            logger.info(f"Loaded {len(self.api_keys)} API keys")
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error loading API keys: {e}")
    
    def _save_api_keys(self) -> None:
        """Save API keys to file storage."""
        keys_file = self.storage_path / "api_keys.json"
        try:
            data = {
                key: api_key.to_dict(include_sensitive=True)
                for key, api_key in self.api_keys.items()
            }
            with open(keys_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug("API keys saved")
        except IOError as e:
            logger.error(f"Error saving API keys: {e}")
            raise
    
    def create_user(self, username: str, email: str, password: str, role: Role) -> User:
        """
        Create a new user.
        
        Args:
            username: Unique username
            email: User email
            password: Plain text password (will be hashed)
            role: User role
            
        Returns:
            Created User object
        """
        if username in self.users:
            raise ValueError(f"User {username} already exists")
        
        password_hash = self._hash_password(password)
        user = User(
            username=username,
            email=email,
            role=role,
            password_hash=password_hash,
            created_at=datetime.now(),
        )
        
        self.users[username] = user
        self._save_users()
        
        logger.info(f"User created: {username} with role {role.value}")
        return user
    
    def authenticate(self, username: str, password: str) -> Optional[User]:
        """
        Verify user credentials.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            User object if authenticated, None otherwise
        """
        if username not in self.users:
            logger.warning(f"Authentication failed: user {username} not found")
            return None
        
        user = self.users[username]
        
        if not user.active:
            logger.warning(f"Authentication failed: user {username} is inactive")
            return None
        
        # Verify password using bcrypt
        if not self._verify_password(password, user.password_hash):
            logger.warning(f"Authentication failed: invalid password for {username}")
            return None
        
        # Update last login
        user.last_login = datetime.now()
        self._save_users()
        
        logger.info(f"User authenticated: {username}")
        return user
    
    def authorize_operation(self, user: User, operation: Permission) -> bool:
        """
        Check if user has permission for operation.
        
        Args:
            user: User object
            operation: Permission to check
            
        Returns:
            True if authorized
        """
        if not user.active:
            logger.warning(f"Authorization denied: user {user.username} is inactive")
            return False
        
        has_perm = user.has_permission(operation)
        
        if has_perm:
            logger.debug(f"User {user.username} authorized for {operation.value}")
        else:
            logger.warning(f"User {user.username} denied for {operation.value}")
        
        return has_perm
    
    def create_api_key(self, user: User, expires_in: Optional[timedelta] = None) -> str:
        """
        Generate time-limited API key.
        
        Args:
            user: User to create key for
            expires_in: Optional expiration duration (default: 90 days)
            
        Returns:
            API key string
        """
        if expires_in is None:
            expires_in = timedelta(days=90)
        
        # Generate secure random key
        key = secrets.token_urlsafe(32)
        
        created_at = datetime.now()
        expires_at = created_at + expires_in
        
        api_key = APIKey(
            key=key,
            username=user.username,
            created_at=created_at,
            expires_at=expires_at,
        )
        
        self.api_keys[key] = api_key
        self._save_api_keys()
        
        logger.info(f"API key created for user {user.username}, expires {expires_at}")
        return key
    
    def verify_api_key(self, key: str) -> Optional[User]:
        """
        Verify API key and return associated user.
        
        Args:
            key: API key to verify
            
        Returns:
            User object if key is valid, None otherwise
        """
        if key not in self.api_keys:
            logger.warning("API key not found")
            return None
        
        api_key = self.api_keys[key]
        
        if not api_key.is_valid():
            logger.warning(f"API key invalid or expired for user {api_key.username}")
            return None
        
        # Update last used timestamp
        api_key.last_used = datetime.now()
        self._save_api_keys()
        
        # Return associated user
        user = self.users.get(api_key.username)
        if user and user.active:
            logger.debug(f"API key verified for user {user.username}")
            return user
        
        logger.warning(f"User {api_key.username} not found or inactive")
        return None
    
    def revoke_api_key(self, key: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            key: API key to revoke
            
        Returns:
            True if revoked successfully
        """
        if key not in self.api_keys:
            logger.warning("API key not found for revocation")
            return False
        
        self.api_keys[key].active = False
        self._save_api_keys()
        
        logger.info(f"API key revoked for user {self.api_keys[key].username}")
        return True
    
    def create_session(self, user: User, expires_in: Optional[timedelta] = None) -> str:
        """
        Create user session.
        
        Args:
            user: User to create session for
            expires_in: Optional session duration (default: 24 hours)
            
        Returns:
            Session ID
        """
        if expires_in is None:
            expires_in = timedelta(hours=24)
        
        session_id = secrets.token_urlsafe(32)
        created_at = datetime.now()
        
        session = Session(
            session_id=session_id,
            username=user.username,
            created_at=created_at,
            expires_at=created_at + expires_in,
            last_activity=created_at,
        )
        
        self.sessions[session_id] = session
        
        logger.info(f"Session created for user {user.username}")
        return session_id
    
    def verify_session(self, session_id: str) -> Optional[User]:
        """
        Verify session and return associated user.
        
        Args:
            session_id: Session ID to verify
            
        Returns:
            User object if session is valid, None otherwise
        """
        if session_id not in self.sessions:
            logger.warning("Session not found")
            return None
        
        session = self.sessions[session_id]
        
        if not session.is_valid():
            logger.warning(f"Session expired for user {session.username}")
            del self.sessions[session_id]
            return None
        
        # Update last activity
        session.last_activity = datetime.now()
        
        # Return associated user
        user = self.users.get(session.username)
        if user and user.active:
            logger.debug(f"Session verified for user {user.username}")
            return user
        
        logger.warning(f"User {session.username} not found or inactive")
        return None
    
    def end_session(self, session_id: str) -> bool:
        """
        End a user session.
        
        Args:
            session_id: Session ID to end
            
        Returns:
            True if session ended successfully
        """
        if session_id not in self.sessions:
            logger.warning("Session not found for termination")
            return False
        
        username = self.sessions[session_id].username
        del self.sessions[session_id]
        
        logger.info(f"Session ended for user {username}")
        return True
    
    def list_users(self) -> List[User]:
        """Get list of all users."""
        return list(self.users.values())
    
    def get_user(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self.users.get(username)
    
    def update_user_role(self, username: str, new_role: Role) -> bool:
        """
        Update user role.
        
        Args:
            username: Username to update
            new_role: New role
            
        Returns:
            True if updated successfully
        """
        if username not in self.users:
            logger.warning(f"User {username} not found for role update")
            return False
        
        old_role = self.users[username].role
        self.users[username].role = new_role
        self._save_users()
        
        logger.info(f"User {username} role updated from {old_role.value} to {new_role.value}")
        return True
    
    def deactivate_user(self, username: str) -> bool:
        """
        Deactivate a user account.
        
        Args:
            username: Username to deactivate
            
        Returns:
            True if deactivated successfully
        """
        if username not in self.users:
            logger.warning(f"User {username} not found for deactivation")
            return False
        
        self.users[username].active = False
        self._save_users()
        
        logger.info(f"User {username} deactivated")
        return True
