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
        if type(self.iat) is not int or type(self.exp) is not int:
            raise PayloadError("iat and exp must be integers")
        if self.exp < self.iat:
            raise PayloadError("exp is lower than iat")
        if not isinstance(self.sub, str) or not self.sub:
            raise PayloadError("sub must be a non-empty string")
        if not isinstance(self.jti, str) or not self.jti:
            raise PayloadError("jti must be a non-empty string")
        if not isinstance(self.iss, str) or not self.iss:
            raise PayloadError("iss must be a non-empty string")
        if not isinstance(self.aud, str) or not self.aud:
            raise PayloadError("aud must be a non-empty string")
        if self.claims is not None and not isinstance(self.claims, dict):
            raise PayloadError("claims must be a dict or None")

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
        if not isinstance(data, dict):
            raise PayloadError("Payload data must be a dictionary")

        required_keys = {
            "jti",
            "sub",
            "iat",
            "exp",
            "iss",
            "aud",
            "claims",
            "token_type",
        }
        missing_keys = required_keys - data.keys()
        if missing_keys:
            raise PayloadError(f"Missing required keys in payload: {missing_keys}")

        jti = data["jti"]
        if not isinstance(jti, str) or not jti:
            raise PayloadError("Payload 'jti' must be a non-empty string")

        sub = data["sub"]
        if not isinstance(sub, str) or not sub:
            raise PayloadError("Payload 'sub' must be a non-empty string")

        iat = data["iat"]
        if type(iat) is not int:
            raise PayloadError("Payload 'iat' must be an integer")

        exp = data["exp"]
        if type(exp) is not int:
            raise PayloadError("Payload 'exp' must be an integer")

        iss = data["iss"]
        if not isinstance(iss, str) or not iss:
            raise PayloadError("Payload 'iss' must be a non-empty string")

        aud = data["aud"]
        if not isinstance(aud, str) or not aud:
            raise PayloadError("Payload 'aud' must be a non-empty string")

        claims = data["claims"]
        if claims is not None and not isinstance(claims, dict):
            raise PayloadError("Payload 'claims' must be a dictionary or None")

        token_type = data["token_type"]
        if token_type not in {"access", "refresh"}:
            raise PayloadError(f"Invalid token type: {token_type}")

        return cls(
            jti=jti,
            sub=sub,
            iat=iat,
            exp=exp,
            iss=iss,
            aud=aud,
            claims=claims,
            token_type=token_type,
        )
