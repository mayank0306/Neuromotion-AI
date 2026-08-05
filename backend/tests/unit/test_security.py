"""Unit tests for security utilities."""

import pytest
from datetime import datetime, timedelta

from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    verify_token,
)


class TestPasswordHashing:
    """Tests for password hashing and verification."""
    
    def test_hash_password_returns_hashed_value(self):
        password = "SecurePassword123!"
        hashed = hash_password(password)
        
        assert hashed != password
        assert len(hashed) > 20

    def test_verify_password_correct(self):
        password = "SecurePassword123!"
        hashed = hash_password(password)
        
        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        password = "SecurePassword123!"
        hashed = hash_password(password)
        
        assert verify_password("WrongPassword123!", hashed) is False

    def test_hash_different_passwords_differently(self):
        password1 = "PasswordOne123!"
        password2 = "PasswordTwo456!"
        
        hash1 = hash_password(password1)
        hash2 = hash_password(password2)
        
        assert hash1 != hash2


class TestTokenManagement:
    """Tests for JWT token creation and validation."""
    
    def test_create_access_token(self):
        data = {"sub": "user-123", "email": "test@example.com"}
        token = create_access_token(data, expires_delta=timedelta(minutes=30))
        
        assert token != ""
        assert len(token.split(".")) == 3  # JWT has 3 parts

    def test_create_refresh_token(self):
        data = {"sub": "user-123"}
        token = create_refresh_token(data, expires_delta=timedelta(days=7))
        
        assert token != ""
        assert len(token.split(".")) == 3

    def test_verify_valid_token(self):
        data = {"sub": "user-123", "email": "test@example.com"}
        token = create_access_token(data, expires_delta=timedelta(minutes=30))
        
        payload = verify_token(token)
        
        assert payload["sub"] == "user-123"
        assert payload["email"] == "test@example.com"
        assert payload["type"] == "access"

    def test_verify_invalid_token_raises_error(self):
        with pytest.raises(Exception):
            verify_token("invalid.jwt.token")

    def test_expired_token_raises_error(self):
        data = {"sub": "user-123"}
        token = create_access_token(data, expires_delta=timedelta(seconds=-1))
        
        with pytest.raises(Exception):
            verify_token(token)
