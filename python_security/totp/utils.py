"""
Module: python_security.totp.utils

Description:
    - Helper Methods for the TOTP
"""

import base64


def _decode_secret(secret: str) -> bytes:
    """
    Restores missing Base32 padding and decodes secret string into bytes.
    """
    if not isinstance(secret, str) or not secret.isascii():
        raise ValueError("Invalid Base32 secret key.")

    cleaned_secret = secret.strip().replace(" ", "").upper()
    missing_padding = len(cleaned_secret) % 8
    if missing_padding:
        cleaned_secret += "=" * (8 - missing_padding)

    try:
        return base64.b32decode(cleaned_secret, casefold=True)
    except Exception as exc:
        raise ValueError("Invalid Base32 secret key.") from exc
