"""Unit tests for input validators."""

import pytest

from app.utils.validators import (
    validate_email,
    validate_password,
    validate_phone,
    validate_height,
    validate_weight,
)


class TestEmailValidation:
    def test_valid_email(self):
        assert validate_email("user@example.com") is True
        assert validate_email("user.name+tag@example.co.uk") is True

    def test_invalid_email(self):
        assert validate_email("not-an-email") is False
        assert validate_email("@example.com") is False
        assert validate_email("user@") is False


class TestPasswordValidation:
    def test_valid_password(self):
        is_valid, msg = validate_password("SecurePass123!@#")
        assert is_valid is True
        assert msg == ""

    def test_short_password(self):
        is_valid, msg = validate_password("Short1!")
        assert is_valid is False
        assert "12 characters" in msg

    def test_no_uppercase(self):
        is_valid, msg = validate_password("lowercase123!@#")
        assert is_valid is False
        assert "uppercase" in msg

    def test_no_lowercase(self):
        is_valid, msg = validate_password("UPPERCASE123!@#")
        assert is_valid is False
        assert "lowercase" in msg

    def test_no_number(self):
        is_valid, msg = validate_password("NoNumbers!@#abc")
        assert is_valid is False
        assert "number" in msg

    def test_no_special_char(self):
        is_valid, msg = validate_password("NoSpecial123")
        assert is_valid is False
        assert "special" in msg.lower()
