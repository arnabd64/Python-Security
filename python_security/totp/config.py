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


@dataclass(frozen=True)
class TOTPSettings:
    num_bytes: int
    algorithm: Literal["SHA1", "SHA256", "SHA512"]
    digits: Literal[6, 8]
    interval: int
    window: int

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
        if self.interval < 10:
            raise InvalidSettings("OTP TTL must be atleast 10 seconds")
        if self.window <= 0 or self.window > 10:
            raise InvalidSettings("window field should be positive integer")

    @classmethod
    def default(cls) -> "TOTPSettings":
        return cls(num_bytes=20, algorithm="SHA1", digits=6, interval=30, window=1)
