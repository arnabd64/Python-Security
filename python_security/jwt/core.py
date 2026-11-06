"""
Module: python_security.jwt.core

Description:
    - Public facing APIs for issueing and verifying JWTs
"""

import time
from typing import Literal

from python_security.exceptions import (
    ExpiredTokenError,
    JWTException,
    SignatureError,
    TokenDecodeError,
)
from python_security.jwt import utils
from python_security.jwt.config import DEFAULT_SETTINGS, JWTSettings
from python_security.jwt.payload import Payload


def issue_access_token(
    sub: str,
    claims: dict[str, str | int | float],
    settings: JWTSettings = DEFAULT_SETTINGS,
) -> tuple[str, str]:
    """
    Issues an Access token against a payload

    Args:
        - `sub`: Subject field
        - `claims`: custom claims

    Returns:
        - `token` (str): Access Token
        - `jti` (str): Unique ID of the JWT
    """
    # time stamp
    now: int = int(time.time())

    # build payload
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
    claims: dict[str, str | int | float],
    settings: JWTSettings = DEFAULT_SETTINGS,
) -> tuple[str, str]:
    """
    Issues an Refresh token against a payload

    Args:
        - `sub`: Subject field
        - `claims`: custom claims

    Returns:
        - `token` (str): Refresh Token
        - `jti` (str): Unique ID of the JWT
    """
    # time stamp
    now: int = int(time.time())

    # build payload
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
    settings: JWTSettings = DEFAULT_SETTINGS,
) -> Payload:
    """
    Decodes a JWT token and returns the payload

    Args:
        - `token` (str): JWT token

    Returns:
        - `payload` (Payload): Instance of Payload
    """
    # check if token is string
    if not isinstance(token, str):
        raise TokenDecodeError("JWT token is not a String")

    # split the token
    parts: list[str] = token.split(".")
    if len(parts) != 3:
        raise TokenDecodeError("Malformed Token")

    # verify signature
    if not utils.verify_signature(parts[0], parts[1], parts[2], settings.secret):
        raise SignatureError

    # decode payload
    payload = Payload.from_dict(utils.json_decode(utils.b64_decode(parts[1])))

    # check expiry
    now: int = int(time.time())
    if payload.exp <= now:
        raise ExpiredTokenError

    # check iss or aud
    if payload.iss != settings.iss or payload.aud != settings.aud:
        raise JWTException("Mismatch is `iss` or `aud` field")

    # verify token type
    if payload.token_type != token_type:
        raise JWTException("Mismatch `token_type`")

    return payload


def verify_token(
    token: str,
    token_type: Literal["access", "refresh"] = "access",
    settings: JWTSettings = DEFAULT_SETTINGS,
) -> bool:
    try:
        _ = parse_token(token, token_type, settings)
        return True

    except JWTException:
        return False


def reissue_access_token(
    refresh_token: str, settings: JWTSettings = DEFAULT_SETTINGS
) -> tuple[str, str]:
    """
    Reissues a new token on sibmitting a refresh token

    Args:
        - `refresh_token` (str): Refresh Token
        - `settings`

    Returns:
        - `token`
        - `jti`
    """
    payload = parse_token(refresh_token, token_type="refresh")

    # timestamp
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
