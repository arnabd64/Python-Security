"""
Module: python_security.jwt.utils

Description:
    - Helper functions
"""

import base64
import hashlib
import hmac
import json
import uuid

from python_security.exceptions import (
    PayloadError,
    TokenDecodeError,
)
from python_security.jwt.config import JWTSettings
from python_security.jwt.payload import Payload

_B64URL_ALPHABET = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
)


def generate_jti() -> str:
    # uuid7 exists on Python 3.14+; older versions raise AttributeError
    generator = getattr(uuid, "uuid7", uuid.uuid4)
    return generator().hex


def json_encode(data: dict) -> bytes:
    try:
        return json.dumps(
            data, separators=(",", ":"), allow_nan=False, ensure_ascii=True
        ).encode("utf-8")
    except (ValueError, TypeError) as e:
        raise ValueError("Invalid JSON data for encoding") from e


def _reject_constant(val: str):
    raise ValueError(f"Non-finite numbers are not allowed: {val}")


def json_decode(data: bytes | str) -> dict:
    try:
        res = json.loads(data, parse_constant=_reject_constant)
        if not isinstance(res, dict):
            raise ValueError("Decoded JSON must be a dictionary")  # noqa: TRY004
        return res
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
        raise ValueError("Failed to decode JSON") from e


def b64_encode(data: bytes) -> str:
    """Unpadded base64url, as required by RFC 7515."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def b64_decode(data: str) -> bytes:
    """Decode unpadded (or padded) base64url into raw bytes strictly."""
    if not isinstance(data, str):
        raise ValueError("Base64 input must be a string")  # noqa: TRY004

    unpadded = data.rstrip("=")
    padding_len = len(data) - len(unpadded)

    if padding_len > 2:
        raise ValueError("Invalid padding length")
    if len(unpadded) % 4 == 1:
        raise ValueError("Impossible base64 string length")

    for c in unpadded:
        if c not in _B64URL_ALPHABET:
            raise ValueError(f"Invalid character in base64url: {c!r}")

    if padding_len > 0:
        if (len(unpadded) + padding_len) % 4 != 0:
            raise ValueError("Invalid base64 padding")
        if data != unpadded + ("=" * padding_len):
            raise ValueError("Padding characters must only be at the end")

    padded = unpadded + "=" * (-len(unpadded) % 4)
    try:
        decoded = base64.urlsafe_b64decode(padded)
    except Exception as e:
        raise ValueError("Base64 decoding failed") from e

    canonical = base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii")
    if canonical != unpadded:
        raise ValueError("Non-canonical base64url encoding")

    return decoded


def sign(b64_header: str, b64_payload: str, secret: str) -> bytes:
    return hmac.new(
        key=secret.encode("utf-8"),
        msg=f"{b64_header}.{b64_payload}".encode(),
        digestmod=hashlib.sha256,
    ).digest()


def issue_token(payload: Payload, settings: JWTSettings) -> tuple[str, str]:
    """
    Issues a Token against a Payload

    Args:
        - `payload` (Payload):  Instance of a Payload

    Returns:
        - `token` (str): Access Token
        - `jti` (str): Unique ID of the JWT
    """
    # Convert the header and payload to JSON String
    b64_header: str = b64_encode(json_encode(settings.header))
    b64_payload: str = b64_encode(json_encode(payload.to_dict()))

    # sign the header & payload
    b64_signature: str = b64_encode(sign(b64_header, b64_payload, settings.secret))

    # deliverables
    token: str = f"{b64_header}.{b64_payload}.{b64_signature}"
    jti: str = payload.jti

    return token, jti


def decode_token(
    b64_header: str, b64_payload: str, b64_signature: str
) -> tuple[dict, Payload, bytes]:
    try:
        header = json_decode(b64_decode(b64_header))
        payload = Payload.from_dict(json_decode(b64_decode(b64_payload)))
        signature = b64_decode(b64_signature)
        return header, payload, signature
    except (ValueError, KeyError, TypeError, PayloadError) as e:
        raise TokenDecodeError("Failed to decode token segments") from e


def verify_signature(
    b64_header: str, b64_payload: str, b64_signature: str, secret: str
) -> bool:
    try:
        if not isinstance(b64_signature, str):
            return False
        token_signature: bytes = b64_decode(b64_signature)
        if len(token_signature) != 32:
            return False
        acquired_sign: bytes = sign(b64_header, b64_payload, secret)
        return hmac.compare_digest(acquired_sign, token_signature)
    except Exception:  # noqa: BLE001
        return False
