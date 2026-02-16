"""
Ethics Module for The Crucible.

This module provides ethical guardrails for adversarial testing including:
- Ethics enforcement and authorization
- Model owner consent management
- Comprehensive audit logging
- Rate limiting and usage tracking
"""

from crucible.ethics.guardrails import (
    EthicsEnforcer,
    ConsentManager,
    AuditLogger,
    ProhibitedUseCase,
    User,
    AuditEntry,
    ConsentRecord,
)

__all__ = [
    "EthicsEnforcer",
    "ConsentManager",
    "AuditLogger",
    "ProhibitedUseCase",
    "User",
    "AuditEntry",
    "ConsentRecord",
]
