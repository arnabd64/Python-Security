"""
Module: python_security.jwt.payload

Description:
    - Schema defintion for Token Payloads
"""

from dataclasses import dataclass
from typing import Literal

from python_security.exceptions import PayloadError


@dataclass(frozen=True, slots=True)
class Payload:
    jti: str
    sub: str
    iat: int
    exp: int
    iss: str
    aud: str
    claims: dict | None
    token_type: Literal["access", "refresh"]

    def __post_init__(self) -> None:
        if self.token_type not in {"access", "refresh"}:
            raise PayloadError(f"Invalid token type: {self.token_type}")
        if self.exp < self.iat:
            raise PayloadError("exp is lower than iat")

    def to_dict(self) -> dict:
        return {
            "jti": self.jti,
            "sub": self.sub,
            "iat": self.iat,
            "exp": self.exp,
            "iss": self.iss,
            "aud": self.aud,
            "claims": self.claims,
            "token_type": self.token_type,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Payload":
        return cls(
            jti=str(data["jti"]),
            sub=data["sub"],
            iat=int(data["iat"]),
            exp=int(data["exp"]),
            iss=data["iss"],
            aud=data["aud"],
            claims=data["claims"],
            token_type=data["token_type"],
        )
