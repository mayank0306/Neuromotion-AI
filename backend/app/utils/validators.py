"""Input validation utilities."""

import re
from typing import Optional


def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate.
        
    Returns:
        True if valid, False otherwise.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_password(password: str) -> tuple[bool, str]:
    """
    Validate password against security requirements.
    
    Requirements:
    - Minimum 12 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one number
    - At least one special character
    
    Args:
        password: Password to validate.
        
    Returns:
        Tuple of (is_valid, error_message).
    """
    if len(password) < 12:
        return False, "Password must be at least 12 characters long"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character"
    
    return True, ""


def validate_phone(phone: str) -> bool:
    """
    Validate phone number format.
    
    Args:
        phone: Phone number to validate.
        
    Returns:
        True if valid, False otherwise.
    """
    # Remove all non-digits
    digits = re.sub(r'\D', '', phone)
    return len(digits) >= 10 and len(digits) <= 15


def validate_height(height: str) -> Optional[float]:
    """
    Validate and parse height string.
    
    Args:
        height: Height string (e.g., "5'10\"", "178cm", "1.78m").
        
    Returns:
        Height in centimeters, or None if invalid.
    """
    if not height:
        return None
    
    # Match feet/inches format
    match = re.match(r"^(\d+)'(\d+)\"$", height)
    if match:
        feet, inches = map(int, match.groups())
        return round((feet * 12 + inches) * 2.54, 1)
    
    # Match cm format
    match = re.match(r"^(\d+(?:\.\d+)?)\s*cm$", height, re.IGNORECASE)
    if match:
        return float(match.group(1))
    
    # Match meters format
    match = re.match(r"^(\d+(?:\.\d+)?)\s*m$", height, re.IGNORECASE)
    if match:
        return float(match.group(1)) * 100
    
    return None


def validate_weight(weight: str) -> Optional[float]:
    """
    Validate and parse weight string.
    
    Args:
        weight: Weight string (e.g., "154lb", "70kg").
        
    Returns:
        Weight in kilograms, or None if invalid.
    """
    if not weight:
        return None
    
    # Match pounds format
    match = re.match(r"^(\d+(?:\.\d+)?)\s*(?:lb|lbs| pounds?)$", weight, re.IGNORECASE)
    if match:
        return round(float(match.group(1)) / 2.205, 1)
    
    # Match kg format
    match = re.match(r"^(\d+(?:\.\d+)?)\s*(?:kg|kilograms?)$", weight, re.IGNORECASE)
    if match:
        return float(match.group(1))
    
    return None
