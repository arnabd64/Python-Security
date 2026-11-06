"""
Module: python_security.jwt.config

Description:
    - Settings for JWT
"""

import os
import secrets
from dataclasses import dataclass

from python_security.exceptions import InvalidSettings

MIN_SECRET_LENGTH = 32
MAX_SECRET_LENGTH = 128


@dataclass(frozen=True, slots=True)
class JWTSettings:
    secret: str
    access_ttl: int
    refresh_ttl: int
    iss: str
    aud: str

    @property
    def header(self) -> dict[str, str]:
        return {"alg": "HS256", "typ": "JWT"}

    def __post_init__(self) -> None:
        if not isinstance(self.secret, str):
            InvalidSettings("JWT secret is not a string")
        if not (MIN_SECRET_LENGTH <= len(self.secret) <= MAX_SECRET_LENGTH):
            raise InvalidSettings(
                f"JWT Secret too short. Must be within: [{MIN_SECRET_LENGTH}, {MAX_SECRET_LENGTH}]"
            )
        if self.refresh_ttl <= 0 or self.access_ttl <= 0:
            raise InvalidSettings("TTL cannot be negative")
        if self.refresh_ttl < self.access_ttl:
            raise InvalidSettings("Refresh TTL is shorter than Access TTL")


DEFAULT_SETTINGS = JWTSettings(
    secret=os.getenv("JWT_SECRET") or secrets.token_hex(32),
    access_ttl=int(os.getenv("JWT_ACCESS_TTL") or (15 * 60)),
    refresh_ttl=int(os.getenv("JWT_REFRESH_TTL") or (5 * 24 * 3600)),
    iss=os.getenv("JWT_ISSUER", "JWTService"),
    aud=os.getenv("JWT_AUDIENCE", "All"),
)
