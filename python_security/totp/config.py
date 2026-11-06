"""
Module: python_security.totp.config

Description:
    - Settings for TOTP
"""

import hashlib
from dataclasses import dataclass
from typing import Literal

from python_security.exceptions import InvalidSettings

ALGORITHMS = {"SHA1": hashlib.sha1, "SHA256": hashlib.sha256, "SHA512": hashlib.sha512}


@dataclass
class TOTPSettings:
    num_bytes: int
    algorithm: Literal["SHA1", "SHA256", "SHA512"]
    digits: Literal[6, 8]
    otp_ttl: int

    @property
    def hasher(self):
        return ALGORITHMS[self.algorithm.upper()]

    def __post_init__(self) -> None:
        if self.num_bytes < 16:
            raise InvalidSettings("num_bytes must be atleast 16 (128 bits)")
        if self.algorithm.upper() not in {"SHA1", "SHA256", "SHA512"}:
            raise InvalidSettings(f"invalid algorithm: {self.algorithm}")
        if self.digits not in {6, 8}:
            raise InvalidSettings(f"invalid number of digits: {self.digits}")
        if self.otp_ttl <= 10:
            raise InvalidSettings("OTP TTL must be atleast 10 seconds")


DEFAULT_SETTINGS = TOTPSettings(num_bytes=20, algorithm="SHA1", digits=6, otp_ttl=30)
