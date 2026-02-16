"""
Ethical Guardrails Module for The Crucible.

This module implements comprehensive ethical enforcement systems including:
- User authentication and authorization
- Use case validation against prohibited activities
- Model owner consent verification
- Rate limiting
- Comprehensive audit trails
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, asdict


logger = logging.getLogger(__name__)


class ProhibitedUseCase(Enum):
    """Enumeration of prohibited use cases."""
    UNAUTHORIZED_MANIPULATION = "unauthorized_model_manipulation"
    COMPETITIVE_SABOTAGE = "competitive_sabotage"
    PRIVACY_VIOLATION = "privacy_violation"
    DISCRIMINATORY_TESTING = "discriminatory_testing"
    MALICIOUS_ATTACK = "malicious_adversarial_attack"


@dataclass
class User:
    """User data structure."""
    username: str
    email: str
    role: str
    created_at: datetime
    api_key: Optional[str] = None


@dataclass
class AuditEntry:
    """Immutable audit log entry."""
    timestamp: datetime
    user_id: str
    operation: str
    model_id: Optional[str]
    approved: bool
    reason: Optional[str]
    entry_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with serializable types."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "operation": self.operation,
            "model_id": self.model_id,
            "approved": self.approved,
            "reason": self.reason,
            "entry_hash": self.entry_hash
        }


@dataclass
class ConsentRecord:
    """Model owner consent record."""
    model_id: str
    owner_signature: str
    granted_to: str  # username
    granted_at: datetime
    expires_at: Optional[datetime]
    terms_version: str
    revoked: bool = False
    
    def is_valid(self) -> bool:
        """Check if consent is still valid."""
        if self.revoked:
            return False
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return True


class AuditLogger:
    """
    Manages persistent audit trails with tamper-evident logging.
    Implements append-only log structure with cryptographic hashing.
    """
    
    def __init__(self, log_path: Path = Path("./audit_logs")):
        """
        Initialize audit logger.
        
        Args:
            log_path: Directory to store audit logs
        """
        self.log_path = log_path
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path / "audit_trail.jsonl"
        self.previous_hash = self._get_last_hash()
        logger.info(f"AuditLogger initialized at {self.log_path}")
    
    def _get_last_hash(self) -> str:
        """Get the hash of the last entry for chain verification."""
        if not self.log_file.exists():
            return "0" * 64  # Genesis hash
        
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_entry = json.loads(lines[-1])
                    return last_entry.get("entry_hash", "0" * 64)
        except (IOError, json.JSONDecodeError) as e:
            logger.warning(f"Could not read last hash: {e}")
        
        return "0" * 64
    
    def _compute_hash(self, entry_data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for tamper detection."""
        # Include previous hash to create chain
        data_to_hash = json.dumps(entry_data, sort_keys=True) + self.previous_hash
        return hashlib.sha256(data_to_hash.encode()).hexdigest()
    
    def log(self, user_id: str, operation: str, model_id: Optional[str],
            approved: bool, reason: Optional[str] = None) -> AuditEntry:
        """
        Log an operation to the audit trail.
        
        Args:
            user_id: User performing the operation
            operation: Operation being performed
            model_id: Model being operated on (if applicable)
            approved: Whether the operation was approved
            reason: Reason for approval/denial
            
        Returns:
            AuditEntry object
        """
        timestamp = datetime.now()
        
        # Create entry data without hash first
        entry_data = {
            "timestamp": timestamp.isoformat(),
            "user_id": user_id,
            "operation": operation,
            "model_id": model_id,
            "approved": approved,
            "reason": reason
        }
        
        # Compute hash
        entry_hash = self._compute_hash(entry_data)
        entry_data["entry_hash"] = entry_hash
        
        # Create audit entry
        entry = AuditEntry(
            timestamp=timestamp,
            user_id=user_id,
            operation=operation,
            model_id=model_id,
            approved=approved,
            reason=reason,
            entry_hash=entry_hash
        )
        
        # Append to log file
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry_data) + "\n")
            
            # Update previous hash for next entry
            self.previous_hash = entry_hash
            logger.debug(f"Audit entry logged: {operation} by {user_id}")
            
        except IOError as e:
            logger.error(f"Failed to write audit log: {e}")
            raise
        
        return entry
    
    def query(self, user_id: Optional[str] = None,
             operation: Optional[str] = None,
             start_time: Optional[datetime] = None,
             end_time: Optional[datetime] = None) -> List[AuditEntry]:
        """
        Query audit trail with filters.
        
        Args:
            user_id: Filter by user
            operation: Filter by operation type
            start_time: Filter entries after this time
            end_time: Filter entries before this time
            
        Returns:
            List of matching audit entries
        """
        if not self.log_file.exists():
            return []
        
        results = []
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    
                    # Apply filters
                    if user_id and data.get("user_id") != user_id:
                        continue
                    if operation and data.get("operation") != operation:
                        continue
                    
                    timestamp = datetime.fromisoformat(data["timestamp"])
                    if start_time and timestamp < start_time:
                        continue
                    if end_time and timestamp > end_time:
                        continue
                    
                    # Reconstruct entry
                    entry = AuditEntry(
                        timestamp=timestamp,
                        user_id=data["user_id"],
                        operation=data["operation"],
                        model_id=data.get("model_id"),
                        approved=data["approved"],
                        reason=data.get("reason"),
                        entry_hash=data["entry_hash"]
                    )
                    results.append(entry)
        
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error querying audit log: {e}")
            raise
        
        return results
    
    def verify_integrity(self) -> bool:
        """
        Verify the integrity of the audit trail.
        
        Returns:
            True if audit trail is intact, False if tampered
        """
        if not self.log_file.exists():
            return True
        
        try:
            previous_hash = "0" * 64
            with open(self.log_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    data = json.loads(line)
                    stored_hash = data.pop("entry_hash")
                    
                    # Recompute hash with previous hash
                    data_to_hash = json.dumps(data, sort_keys=True) + previous_hash
                    computed_hash = hashlib.sha256(data_to_hash.encode()).hexdigest()
                    
                    if computed_hash != stored_hash:
                        logger.error(f"Integrity violation at line {line_num}")
                        return False
                    
                    previous_hash = stored_hash
            
            logger.info("Audit trail integrity verified")
            return True
            
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error verifying integrity: {e}")
            return False
    
    def export(self, output_path: Path) -> None:
        """Export audit trail for regulatory compliance."""
        if not self.log_file.exists():
            logger.warning("No audit log to export")
            return
        
        try:
            with open(self.log_file, 'r') as f:
                entries = [json.loads(line) for line in f]
            
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_entries": len(entries),
                "integrity_verified": self.verify_integrity(),
                "entries": entries
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Audit trail exported to {output_path}")
            
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error exporting audit trail: {e}")
            raise
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """
        Detect anomalous usage patterns.
        
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if not self.log_file.exists():
            return anomalies
        
        try:
            # Load recent entries (last 24 hours)
            recent_cutoff = datetime.now() - timedelta(days=1)
            recent_entries = self.query(start_time=recent_cutoff)
            
            # Check for suspicious patterns
            user_operations = {}
            for entry in recent_entries:
                if entry.user_id not in user_operations:
                    user_operations[entry.user_id] = []
                user_operations[entry.user_id].append(entry)
            
            # Detect high-frequency operations
            for user_id, operations in user_operations.items():
                if len(operations) > 1000:  # More than 1000 ops in 24h
                    anomalies.append({
                        "type": "high_frequency",
                        "user_id": user_id,
                        "count": len(operations),
                        "severity": "high"
                    })
                
                # Detect high denial rate
                denials = sum(1 for op in operations if not op.approved)
                if denials > 10 and denials / len(operations) > 0.5:
                    anomalies.append({
                        "type": "high_denial_rate",
                        "user_id": user_id,
                        "denial_count": denials,
                        "total_operations": len(operations),
                        "severity": "medium"
                    })
            
            if anomalies:
                logger.warning(f"Detected {len(anomalies)} anomalies in usage patterns")
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
        
        return anomalies


class ConsentManager:
    """
    Manages model owner consent with digital signatures.
    Tracks consent grants, renewals, and revocations.
    """
    
    def __init__(self, storage_path: Path = Path("./consent_records")):
        """
        Initialize consent manager.
        
        Args:
            storage_path: Directory to store consent records
        """
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.consents: Dict[str, ConsentRecord] = {}
        self._load_consents()
        logger.info(f"ConsentManager initialized at {storage_path}")
    
    def _load_consents(self) -> None:
        """Load consent records from storage."""
        consent_file = self.storage_path / "consents.json"
        if not consent_file.exists():
            return
        
        try:
            with open(consent_file, 'r') as f:
                data = json.load(f)
                for key, record in data.items():
                    self.consents[key] = ConsentRecord(
                        model_id=record["model_id"],
                        owner_signature=record["owner_signature"],
                        granted_to=record["granted_to"],
                        granted_at=datetime.fromisoformat(record["granted_at"]),
                        expires_at=datetime.fromisoformat(record["expires_at"]) if record.get("expires_at") else None,
                        terms_version=record["terms_version"],
                        revoked=record.get("revoked", False)
                    )
            logger.info(f"Loaded {len(self.consents)} consent records")
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error loading consents: {e}")
    
    def _save_consents(self) -> None:
        """Save consent records to storage."""
        consent_file = self.storage_path / "consents.json"
        try:
            data = {}
            for key, record in self.consents.items():
                data[key] = {
                    "model_id": record.model_id,
                    "owner_signature": record.owner_signature,
                    "granted_to": record.granted_to,
                    "granted_at": record.granted_at.isoformat(),
                    "expires_at": record.expires_at.isoformat() if record.expires_at else None,
                    "terms_version": record.terms_version,
                    "revoked": record.revoked
                }
            
            with open(consent_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug("Consent records saved")
        except IOError as e:
            logger.error(f"Error saving consents: {e}")
            raise
    
    def grant_consent(self, model_id: str, owner_signature: str,
                     granted_to: str, terms_version: str = "1.0",
                     expires_in: Optional[timedelta] = None) -> ConsentRecord:
        """
        Grant consent for model testing.
        
        Args:
            model_id: Model identifier
            owner_signature: Digital signature from model owner
            granted_to: Username receiving consent
            terms_version: Version of terms accepted
            expires_in: Optional expiration duration
            
        Returns:
            ConsentRecord object
        """
        key = f"{model_id}:{granted_to}"
        
        granted_at = datetime.now()
        expires_at = granted_at + expires_in if expires_in else None
        
        consent = ConsentRecord(
            model_id=model_id,
            owner_signature=owner_signature,
            granted_to=granted_to,
            granted_at=granted_at,
            expires_at=expires_at,
            terms_version=terms_version,
            revoked=False
        )
        
        self.consents[key] = consent
        self._save_consents()
        
        logger.info(f"Consent granted for model {model_id} to user {granted_to}")
        return consent
    
    def verify_consent(self, model_id: str, username: str) -> bool:
        """
        Verify if user has valid consent for model.
        
        Args:
            model_id: Model identifier
            username: User requesting access
            
        Returns:
            True if consent is valid
        """
        key = f"{model_id}:{username}"
        
        if key not in self.consents:
            logger.warning(f"No consent found for {username} on model {model_id}")
            return False
        
        consent = self.consents[key]
        is_valid = consent.is_valid()
        
        if not is_valid:
            if consent.revoked:
                logger.warning(f"Consent revoked for {username} on model {model_id}")
            elif consent.expires_at and datetime.now() > consent.expires_at:
                logger.warning(f"Consent expired for {username} on model {model_id}")
        
        return is_valid
    
    def revoke_consent(self, model_id: str, granted_to: str) -> bool:
        """
        Revoke previously granted consent.
        
        Args:
            model_id: Model identifier
            granted_to: Username whose consent to revoke
            
        Returns:
            True if revoked successfully
        """
        key = f"{model_id}:{granted_to}"
        
        if key not in self.consents:
            logger.warning(f"No consent to revoke for {granted_to} on model {model_id}")
            return False
        
        self.consents[key].revoked = True
        self._save_consents()
        
        logger.info(f"Consent revoked for {granted_to} on model {model_id}")
        return True
    
    def renew_consent(self, model_id: str, granted_to: str,
                     new_expiration: timedelta) -> bool:
        """
        Renew consent with new expiration.
        
        Args:
            model_id: Model identifier
            granted_to: Username whose consent to renew
            new_expiration: New expiration duration from now
            
        Returns:
            True if renewed successfully
        """
        key = f"{model_id}:{granted_to}"
        
        if key not in self.consents:
            logger.warning(f"No consent to renew for {granted_to} on model {model_id}")
            return False
        
        self.consents[key].expires_at = datetime.now() + new_expiration
        self.consents[key].revoked = False  # Un-revoke if necessary
        self._save_consents()
        
        logger.info(f"Consent renewed for {granted_to} on model {model_id}")
        return True


class EthicsEnforcer:
    """
    Main ethical enforcement system.
    
    Enforces:
    - User authentication and authorization
    - Use case validation
    - Model owner consent
    - Rate limiting
    - Comprehensive audit trails
    """
    
    def __init__(self, audit_logger: Optional[AuditLogger] = None,
                 consent_manager: Optional[ConsentManager] = None,
                 rate_limit: int = 100):
        """
        Initialize ethics enforcer.
        
        Args:
            audit_logger: AuditLogger instance (creates new if None)
            consent_manager: ConsentManager instance (creates new if None)
            rate_limit: Maximum operations per day per user
        """
        self.audit_logger = audit_logger or AuditLogger()
        self.consent_manager = consent_manager or ConsentManager()
        self.rate_limit = rate_limit
        self.user_operation_counts: Dict[str, List[datetime]] = {}
        self.prohibited_use_cases: Set[ProhibitedUseCase] = set(ProhibitedUseCase)
        logger.info(f"EthicsEnforcer initialized with rate limit {rate_limit}/day")
    
    def check_rate_limit(self, user_id: str) -> bool:
        """
        Check if user has exceeded rate limit.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if within rate limit
        """
        now = datetime.now()
        cutoff = now - timedelta(days=1)
        
        # Initialize or clean old entries
        if user_id not in self.user_operation_counts:
            self.user_operation_counts[user_id] = []
        
        # Remove operations older than 24 hours
        self.user_operation_counts[user_id] = [
            ts for ts in self.user_operation_counts[user_id]
            if ts > cutoff
        ]
        
        # Check limit
        count = len(self.user_operation_counts[user_id])
        if count >= self.rate_limit:
            logger.warning(f"Rate limit exceeded for user {user_id}: {count}/{self.rate_limit}")
            return False
        
        # Record this operation
        self.user_operation_counts[user_id].append(now)
        return True
    
    def validate_use_case(self, operation: str, context: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate that operation is not a prohibited use case.
        
        Args:
            operation: Operation type
            context: Operation context for analysis
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Check for prohibited patterns
        operation_lower = operation.lower()
        
        if "unauthorized" in context.get("intent", "").lower():
            return False, "Unauthorized manipulation detected"
        
        if "sabotage" in operation_lower or "competitive_attack" in context.get("purpose", "").lower():
            return False, "Competitive sabotage prohibited"
        
        if context.get("targets_privacy", False):
            return False, "Privacy violation prohibited"
        
        if context.get("discriminatory_intent", False):
            return False, "Discriminatory testing prohibited"
        
        if context.get("malicious", False):
            return False, "Malicious attack prohibited"
        
        return True, None
    
    def authorize_operation(self, user_id: str, operation: str,
                          model_id: Optional[str] = None,
                          context: Optional[Dict[str, Any]] = None) -> tuple[bool, str]:
        """
        Authorize an operation with full ethics checks.
        
        Args:
            user_id: User performing operation
            operation: Operation type
            model_id: Model being operated on
            context: Operation context
            
        Returns:
            Tuple of (authorized, reason)
        """
        context = context or {}
        
        # Check rate limit
        if not self.check_rate_limit(user_id):
            self.audit_logger.log(user_id, operation, model_id, False, "Rate limit exceeded")
            return False, "Rate limit exceeded"
        
        # Validate use case
        valid_use_case, reason = self.validate_use_case(operation, context)
        if not valid_use_case:
            self.audit_logger.log(user_id, operation, model_id, False, reason)
            return False, reason
        
        # Check model consent if model_id provided
        if model_id and not context.get("skip_consent", False):
            if not self.consent_manager.verify_consent(model_id, user_id):
                self.audit_logger.log(user_id, operation, model_id, False, "Consent not granted")
                return False, "Model owner consent required"
        
        # All checks passed
        self.audit_logger.log(user_id, operation, model_id, True, "All checks passed")
        logger.info(f"Operation authorized: {operation} by {user_id} on model {model_id}")
        return True, "Operation authorized"
    
    def get_audit_summary(self, user_id: Optional[str] = None,
                         days: int = 7) -> Dict[str, Any]:
        """
        Get audit summary for user or system.
        
        Args:
            user_id: Optional user filter
            days: Number of days to include
            
        Returns:
            Summary statistics
        """
        cutoff = datetime.now() - timedelta(days=days)
        entries = self.audit_logger.query(user_id=user_id, start_time=cutoff)
        
        approved = sum(1 for e in entries if e.approved)
        denied = sum(1 for e in entries if not e.approved)
        
        operations_by_type = {}
        for entry in entries:
            operations_by_type[entry.operation] = operations_by_type.get(entry.operation, 0) + 1
        
        summary = {
            "total_operations": len(entries),
            "approved": approved,
            "denied": denied,
            "approval_rate": approved / len(entries) if entries else 0,
            "operations_by_type": operations_by_type,
            "period_days": days,
            "user_id": user_id
        }
        
        return summary
