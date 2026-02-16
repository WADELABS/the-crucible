"""
Tests for authentication module with bcrypt password hashing.
"""
import pytest
from pathlib import Path
import tempfile
import json

from crucible.auth.authentication import UserAuthenticator, Role, Permission


class TestBcryptPasswordHashing:
    """Test bcrypt password hashing implementation."""
    
    def test_bcrypt_hash_format(self):
        """Test that passwords are hashed using bcrypt format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            auth = UserAuthenticator(storage_path=Path(tmpdir))
            user = auth.create_user("testuser", "test@example.com", "TestPass123!", Role.ANALYST)
            
            # Bcrypt hashes start with $2b$ and are 60 characters
            assert user.password_hash.startswith('$2b$'), "Hash should use bcrypt format"
            assert len(user.password_hash) == 60, "Bcrypt hash should be 60 characters"
    
    def test_password_authentication_success(self):
        """Test successful authentication with correct password."""
        with tempfile.TemporaryDirectory() as tmpdir:
            auth = UserAuthenticator(storage_path=Path(tmpdir))
            auth.create_user("testuser", "test@example.com", "SecurePass123!", Role.ANALYST)
            
            authenticated = auth.authenticate("testuser", "SecurePass123!")
            assert authenticated is not None, "Authentication should succeed with correct password"
            assert authenticated.username == "testuser"
    
    def test_password_authentication_failure(self):
        """Test authentication fails with wrong password."""
        with tempfile.TemporaryDirectory() as tmpdir:
            auth = UserAuthenticator(storage_path=Path(tmpdir))
            auth.create_user("testuser", "test@example.com", "SecurePass123!", Role.ANALYST)
            
            authenticated = auth.authenticate("testuser", "WrongPassword")
            assert authenticated is None, "Authentication should fail with wrong password"
    
    def test_different_passwords_different_hashes(self):
        """Test that same password for different users produces different hashes (due to salt)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            auth = UserAuthenticator(storage_path=Path(tmpdir))
            user1 = auth.create_user("user1", "user1@example.com", "SamePassword", Role.ANALYST)
            user2 = auth.create_user("user2", "user2@example.com", "SamePassword", Role.ANALYST)
            
            # Even with same password, hashes should be different due to unique salts
            assert user1.password_hash != user2.password_hash, "Same password should produce different hashes"
    
    def test_password_persistence(self):
        """Test that password hashes persist correctly across loads."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)
            
            # Create user
            auth1 = UserAuthenticator(storage_path=storage_path)
            auth1.create_user("testuser", "test@example.com", "TestPass123!", Role.ANALYST)
            
            # Load users from storage
            auth2 = UserAuthenticator(storage_path=storage_path)
            authenticated = auth2.authenticate("testuser", "TestPass123!")
            
            assert authenticated is not None, "Should authenticate after loading from storage"
            assert authenticated.username == "testuser"
    
    def test_password_verification_with_invalid_hash(self):
        """Test that verification handles invalid hash gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            auth = UserAuthenticator(storage_path=Path(tmpdir))
            
            # Test with invalid hash format
            result = auth._verify_password("password", "invalid_hash")
            assert result is False, "Should return False for invalid hash"
    
    def test_case_sensitive_passwords(self):
        """Test that passwords are case-sensitive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            auth = UserAuthenticator(storage_path=Path(tmpdir))
            auth.create_user("testuser", "test@example.com", "Password123", Role.ANALYST)
            
            # Correct password
            assert auth.authenticate("testuser", "Password123") is not None
            
            # Wrong case
            assert auth.authenticate("testuser", "password123") is None
            assert auth.authenticate("testuser", "PASSWORD123") is None
    
    def test_empty_password_handling(self):
        """Test handling of empty passwords."""
        with tempfile.TemporaryDirectory() as tmpdir:
            auth = UserAuthenticator(storage_path=Path(tmpdir))
            
            # Should be able to create user with empty password (not recommended but technically allowed)
            user = auth.create_user("testuser", "test@example.com", "", Role.ANALYST)
            assert user.password_hash.startswith('$2b$')
            
            # Should be able to authenticate with empty password
            assert auth.authenticate("testuser", "") is not None
    
    def test_unicode_password_support(self):
        """Test that unicode passwords are supported."""
        with tempfile.TemporaryDirectory() as tmpdir:
            auth = UserAuthenticator(storage_path=Path(tmpdir))
            unicode_password = "パスワード123!@#"
            
            auth.create_user("testuser", "test@example.com", unicode_password, Role.ANALYST)
            authenticated = auth.authenticate("testuser", unicode_password)
            
            assert authenticated is not None, "Should support unicode passwords"
    
    def test_bcrypt_computational_cost(self):
        """Test that bcrypt hashing is using appropriate work factor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            auth = UserAuthenticator(storage_path=Path(tmpdir))
            user = auth.create_user("testuser", "test@example.com", "TestPass", Role.ANALYST)
            
            # Extract work factor from hash (format: $2b$12$...)
            parts = user.password_hash.split('$')
            work_factor = int(parts[2])
            
            # Should use at least 12 rounds (2^12 iterations)
            assert work_factor >= 12, f"Work factor should be at least 12, got {work_factor}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
