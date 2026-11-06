"""
Module: python_security.totp.core

Description:
    - Public facing functions
"""

import base64
import secrets

from python_security.totp.config import DEFAULT_SETTINGS, TOTPSettings


def new_totp_secret(settings: TOTPSettings = DEFAULT_SETTINGS) -> str:
    """
    Generates a new TOTP Secret
    """
    return (
        base64.b32encode(secrets.token_bytes(settings.num_bytes))
        .decode("ascii")
        .rstrip("=")
    )


def totp_uri() -> str:
    """
    Generates a otpauth://... URI
    """
    ...


def verify_otp() -> bool:
    """
    Verifies an OTP
    """
    ...
