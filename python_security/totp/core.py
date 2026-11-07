"""
Module: python_security.totp.core

Description:
    - Public facing functions
"""

import base64
import hmac
import math
import secrets
import struct
import time
import urllib.parse

from python_security.totp.config import TOTPSettings
from python_security.totp.utils import _decode_secret

# State tracking for replay protection and brute force throttling
_used_counters: set[tuple[bytes, int]] = set()
_failed_attempts: dict[bytes, int] = {}
MAX_FAILED_ATTEMPTS = 20


def reset_state() -> None:
    """Reset replay protection cache and failed attempt counters."""
    _used_counters.clear()
    _failed_attempts.clear()


def new_totp_secret(settings: TOTPSettings = TOTPSettings.default()) -> str:
    """
    Generates a new TOTP Secret
    """
    return (
        base64.b32encode(secrets.token_bytes(settings.num_bytes))
        .decode("ascii")
        .rstrip("=")
    )


def totp_uri(
    secret: str,
    account_name: str,
    issuer: str,
    settings: TOTPSettings = TOTPSettings.default(),
) -> str:
    """
    Generates a otpauth://... URI
    """
    # Validate secret
    _decode_secret(secret)

    # Encode label using colon separator
    label_urlsafe: str = urllib.parse.quote(f"{issuer}:{account_name}", safe=":")

    params = {
        "secret": secret.strip().replace(" ", "").upper(),
        "issuer": issuer,
        "algorithm": settings.algorithm.upper(),
        "digits": settings.digits,
        "period": settings.interval,
    }

    query_string: str = urllib.parse.urlencode(params)

    return f"otpauth://totp/{label_urlsafe}?{query_string}"


def generate_otp(
    secret: str,
    for_time: float | None = None,
    counter_offset: int = 0,
    settings: TOTPSettings = TOTPSettings.default(),
) -> str:
    """
    Generates a TOTP code for a specific time and counter offset.
    """
    key_bytes = _decode_secret(secret)
    if len(key_bytes) < 16:
        raise ValueError("Secret must be at least 16 bytes (128 bits)")

    timestamp = time.time() if for_time is None else for_time
    if (
        not isinstance(timestamp, (int, float))
        or math.isnan(timestamp)
        or math.isinf(timestamp)
    ):
        raise ValueError("Invalid timestamp")
    if timestamp < 0:
        raise ValueError("Timestamp cannot be negative")

    counter = int(timestamp // settings.interval) + counter_offset
    if counter < 0 or counter >= (1 << 64):
        raise ValueError(f"Counter out of 64-bit unsigned range: {counter}")

    counter_bytes = struct.pack(">Q", counter)

    # Compute HMAC
    hmac_digest = hmac.new(key_bytes, counter_bytes, settings.hasher).digest()

    # Dynamic Truncation (RFC 4226)
    offset = hmac_digest[-1] & 0x0F
    binary_code = struct.unpack(">I", hmac_digest[offset : offset + 4])[0] & 0x7FFFFFFF

    code = binary_code % (10**settings.digits)
    return f"{code:0{settings.digits}d}"


def verify_otp(
    otp: str,
    secret: str,
    settings: TOTPSettings = TOTPSettings.default(),
    for_time: float | None = None,
) -> bool:
    """
    Verifies an OTP
    """
    if not isinstance(otp, str):
        return False

    otp = otp.strip()

    if len(otp) != settings.digits or not (otp.isascii() and otp.isdigit()):
        return False

    try:
        secret_bytes = _decode_secret(secret)
    except (ValueError, TypeError):
        return False

    if len(secret_bytes) < 16:
        return False

    if _failed_attempts.get(secret_bytes, 0) >= MAX_FAILED_ATTEMPTS:
        return False

    now: float = time.time() if for_time is None else for_time
    if not isinstance(now, (int, float)) or math.isnan(now) or math.isinf(now):
        return False

    matched = False
    matched_counter = None

    for offset in range(-settings.window, settings.window + 1):
        try:
            expected_otp = generate_otp(
                secret=secret, for_time=now, counter_offset=offset, settings=settings
            )
        except (ValueError, OverflowError):
            continue

        if hmac.compare_digest(expected_otp.encode("ascii"), otp.encode("ascii")):
            curr_counter = int(now // settings.interval) + offset
            if (secret_bytes, curr_counter) not in _used_counters:
                matched = True
                matched_counter = curr_counter

    if matched and matched_counter is not None:
        _used_counters.add((secret_bytes, matched_counter))
        _failed_attempts.pop(secret_bytes, None)
        return True

    _failed_attempts[secret_bytes] = _failed_attempts.get(secret_bytes, 0) + 1
    return False
