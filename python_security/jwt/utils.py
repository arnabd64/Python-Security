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

from python_security.jwt.config import JWTSettings
from python_security.jwt.payload import Payload


def generate_jti() -> str:
    # uuid7 exists on Python 3.14+; older versions raise AttributeError
    generator = getattr(uuid, "uuid7", uuid.uuid4)
    return generator().hex


def json_encode(data: dict) -> bytes:
    return json.dumps(data, separators=(",", ":")).encode("utf-8")


def json_decode(data: bytes) -> dict:
    return json.loads(data)


def b64_encode(data: bytes) -> str:
    """Unpadded base64url, as required by RFC 7515."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def b64_decode(data: str) -> bytes:
    """Decode unpadded (or padded) base64url into raw bytes."""
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def sign(b64_header: str, b64_payload: str, secret: str) -> bytes:
    return hmac.new(
        key=secret.encode(),
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


def decode_token(b64_header: str, b64_payload: str, b64_signature: str):
    header = json_decode(b64_decode(b64_header))
    payload = Payload.from_dict(json_decode(b64_decode(b64_payload)))
    signature = b64_decode(b64_signature)

    return header, payload, signature


def verify_signature(
    b64_header: str, b64_payload: str, b64_signature: str, secret: str
) -> bool:
    # get the sigfnature
    acquired_sign: bytes = sign(b64_header, b64_payload, secret)

    # decode the token signature
    token_signature: bytes = b64_decode(b64_signature)

    return hmac.compare_digest(acquired_sign, token_signature)
