"""
Module: python_security.jwt.config

Description:
    - Settings for JWT
"""

import math
import os
import secrets
from collections import Counter
from dataclasses import dataclass

from python_security.exceptions import InvalidSettings

MIN_SECRET_LENGTH = 32
MAX_SECRET_LENGTH = 64
DEFAULT_ACCESS_TTL = 900  # 15 minutes
DEFAULT_REFRESH_TTL = 432_000  # 5 days
DEFAULT_ISSUER = "JWTService"
DEFAULT_AUDIENCE = "All"


def _is_weak_secret(secret: str) -> bool:
    if (secret + secret).find(secret, 1) < len(secret):
        return True
    counts = Counter(secret)
    total = len(secret)
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return bool(entropy < 3.0 or len(counts) < 8)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        return int(raw)
    except ValueError:
        raise InvalidSettings(f"{name} must be an integer") from None


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
            raise InvalidSettings("JWT secret is not a string")
        if not (MIN_SECRET_LENGTH <= len(self.secret) <= MAX_SECRET_LENGTH):
            raise InvalidSettings(
                f"JWT Secret length is out of range. Must be within: [{MIN_SECRET_LENGTH}, {MAX_SECRET_LENGTH}]"
            )
        if _is_weak_secret(self.secret):
            raise InvalidSettings(
                "JWT secret has insufficient entropy or repeating patterns"
            )
        if type(self.access_ttl) is not int or type(self.refresh_ttl) is not int:
            raise InvalidSettings("TTL must be an integer")
        if self.refresh_ttl <= 0 or self.access_ttl <= 0:
            raise InvalidSettings("TTL cannot be negative or zero")
        if self.refresh_ttl < self.access_ttl:
            raise InvalidSettings("Refresh TTL is shorter than Access TTL")
        if not isinstance(self.iss, str) or not self.iss:
            raise InvalidSettings("Issuer cannot be empty")
        if not isinstance(self.aud, str) or not self.aud:
            raise InvalidSettings("Audience cannot be empty")

    def __repr__(self) -> str:
        return f"JWTSettings(secret=*****, access_ttl={self.access_ttl}, refresh_ttl={self.refresh_ttl}, iss={self.iss}, aud={self.aud})"

    def __str__(self) -> str:
        return self.__repr__()

    @classmethod
    def default(cls) -> "JWTSettings":
        return cls.from_env()

    @classmethod
    def from_env(cls) -> "JWTSettings":
        secret = os.environ.get("JWT_SECRET")
        if not secret:
            raise InvalidSettings("JWT_SECRET environment variable is missing or empty")

        return cls(
            secret=secret,
            access_ttl=_env_int("JWT_ACCESS_TTL", DEFAULT_ACCESS_TTL),
            refresh_ttl=_env_int("JWT_REFRESH_TTL", DEFAULT_REFRESH_TTL),
            iss=os.environ.get("JWT_ISSUER", DEFAULT_ISSUER),
            aud=os.environ.get("JWT_AUDIENCE", DEFAULT_AUDIENCE),
        )
