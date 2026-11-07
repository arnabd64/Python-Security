"""
Module: python_security.jwt.core

Description:
    - Public facing APIs for issueing and verifying JWTs
"""

import math
import time
from typing import Literal

from python_security.exceptions import (
    ExpiredTokenError,
    JWTException,
    PayloadError,
    SignatureError,
    TokenDecodeError,
)
from python_security.jwt import utils
from python_security.jwt.config import JWTSettings
from python_security.jwt.payload import Payload

_B64URL_SET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")


def issue_access_token(
    sub: str,
    claims: dict[str, str | int | float] | None = None,
    settings: JWTSettings | None = None,
) -> tuple[str, str]:
    """
    Issues an Access token against a payload

    Args:
        - `sub`: Subject field
        - `claims`: custom claims
        - `settings`: Optional JWTSettings

    Returns:
        - `token` (str): Access Token
        - `jti` (str): Unique ID of the JWT
    """
    if settings is None:
        settings = JWTSettings.default()

    if not isinstance(sub, str) or not sub:
        raise ValueError("Subject 'sub' must be a non-empty string")

    if claims is None:
        claims = {}
    elif not isinstance(claims, dict):
        raise TypeError("Claims must be a dictionary")

    for k, v in claims.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            raise ValueError(f"Claim {k!r} has non-finite float value")

    now: int = int(time.time())

    payload: Payload = Payload(
        jti=utils.generate_jti(),
        sub=sub,
        claims=claims,
        iat=now,
        exp=now + settings.access_ttl,
        iss=settings.iss,
        aud=settings.aud,
        token_type="access",
    )

    return utils.issue_token(payload, settings)


def issue_refresh_token(
    sub: str,
    claims: dict[str, str | int | float] | None = None,
    settings: JWTSettings | None = None,
) -> tuple[str, str]:
    """
    Issues a Refresh token against a payload

    Args:
        - `sub`: Subject field
        - `claims`: custom claims
        - `settings`: Optional JWTSettings

    Returns:
        - `token` (str): Refresh Token
        - `jti` (str): Unique ID of the JWT
    """
    if settings is None:
        settings = JWTSettings.default()

    if not isinstance(sub, str) or not sub:
        raise ValueError("Subject 'sub' must be a non-empty string")

    if claims is None:
        claims = {}
    elif not isinstance(claims, dict):
        raise TypeError("Claims must be a dictionary")

    for k, v in claims.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            raise ValueError(f"Claim {k!r} has non-finite float value")

    now: int = int(time.time())

    payload: Payload = Payload(
        jti=utils.generate_jti(),
        sub=sub,
        claims=claims,
        iat=now,
        exp=now + settings.refresh_ttl,
        iss=settings.iss,
        aud=settings.aud,
        token_type="refresh",
    )

    return utils.issue_token(payload, settings)


def parse_token(
    token: str,
    token_type: Literal["access", "refresh"] = "access",
    settings: JWTSettings | None = None,
) -> Payload:
    """
    Decodes a JWT token and returns the payload

    Args:
        - `token` (str): JWT token
        - `token_type`: Expected token type ("access" or "refresh")
        - `settings`: Optional JWTSettings

    Returns:
        - `payload` (Payload): Instance of Payload
    """
    if settings is None:
        settings = JWTSettings.default()

    if token_type not in ("access", "refresh"):
        raise ValueError(f"Invalid expected token_type: {token_type!r}")

    if not isinstance(token, str):
        raise TokenDecodeError("JWT token is not a string")

    if token != token.strip() or token.startswith("Bearer "):
        raise TokenDecodeError("Token has invalid formatting or whitespace")

    parts: list[str] = token.split(".")
    if len(parts) != 3:
        raise TokenDecodeError(
            "Malformed token: must have exactly 3 segments separated by '.'"
        )

    b64_header, b64_payload, b64_signature = parts

    if not b64_header or not b64_payload:
        raise TokenDecodeError("Header and payload segments cannot be empty")
    if not b64_signature:
        raise SignatureError("Signature segment cannot be empty")

    # Wire format validation: each segment must strictly be unpadded base64url characters
    for seg in parts:
        if "=" in seg or not set(seg).issubset(_B64URL_SET):
            raise TokenDecodeError("Invalid characters or padding in token segment")

    # Verify signature before decoding payload (prevent parser oracle)
    if not utils.verify_signature(
        b64_header, b64_payload, b64_signature, settings.secret
    ):
        raise SignatureError("Invalid token signature")

    # Decode header
    try:
        header_raw = utils.b64_decode(b64_header)
        header = utils.json_decode(header_raw)
    except Exception as e:
        raise TokenDecodeError("Malformed token header") from e

    if not isinstance(header, dict):
        raise TokenDecodeError("Token header must be a JSON object")

    if header.get("alg") != "HS256":
        raise JWTException("Invalid algorithm in header: expected HS256")
    if header.get("typ") != "JWT":
        raise JWTException("Invalid typ in header: expected JWT")
    if "crit" in header:
        raise JWTException("Critical header parameter 'crit' is not supported")

    # Decode payload
    try:
        payload_raw = utils.b64_decode(b64_payload)
        payload_dict = utils.json_decode(payload_raw)
        payload = Payload.from_dict(payload_dict)
    except PayloadError:
        raise
    except Exception as e:
        raise TokenDecodeError("Malformed token payload") from e

    now: int = int(time.time())

    if payload.iat > now:
        raise JWTException("Token issued in the future (iat > now)")

    if "nbf" in payload_dict:
        nbf = payload_dict["nbf"]
        if not isinstance(nbf, int) or isinstance(nbf, bool):
            raise PayloadError("nbf claim must be an integer")
        if now < nbf:
            raise JWTException("Token not yet valid (nbf > now)")

    if payload.exp <= now:
        raise ExpiredTokenError("Token has expired")

    if payload.iss != settings.iss or payload.aud != settings.aud:
        raise JWTException("Mismatch in iss or aud claim")

    if payload.token_type != token_type:
        raise JWTException(
            f"Mismatch token_type: expected {token_type}, got {payload.token_type}"
        )

    return payload


def verify_token(
    token: str,
    token_type: Literal["access", "refresh"] = "access",
    settings: JWTSettings | None = None,
) -> bool:
    try:
        _ = parse_token(token, token_type, settings)
        return True
    except Exception:  # noqa: BLE001
        return False


def reissue_access_token(
    refresh_token: str, settings: JWTSettings | None = None
) -> tuple[str, str]:
    """
    Reissues a new access token on submitting a valid refresh token

    Args:
        - `refresh_token` (str): Refresh Token
        - `settings`: Optional JWTSettings

    Returns:
        - `token`: New access token
        - `jti`: JTI of new access token
    """
    if settings is None:
        settings = JWTSettings.default()

    payload = parse_token(refresh_token, token_type="refresh", settings=settings)

    now: int = int(time.time())

    new_payload = Payload(
        jti=utils.generate_jti(),
        sub=payload.sub,
        iat=now,
        exp=min(now + settings.access_ttl, payload.exp),
        iss=settings.iss,
        aud=settings.aud,
        claims=payload.claims,
        token_type="access",
    )

    return utils.issue_token(new_payload, settings)
