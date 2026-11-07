"""
Module: python_security.api_key.utils

Description:
    - Helper and Utility functions for the main module
"""

import hashlib
import hmac


def key_digest(data: str) -> bytes:
    return hashlib.sha256(data.encode()).digest()


def verify_digest(secret: str, expected_digest: str) -> bool:
    digest_bytes = key_digest(secret).hex().encode("ascii")
    return hmac.compare_digest(digest_bytes, expected_digest.lower().encode("utf-8"))
