"""
Module: python_security.scrypt.config

Description:
    - Settings needed for SCrypt Password Hashing
"""

from dataclasses import dataclass

from python_security.scrypt.exceptions import InvalidSettingsValue

# Limits
MIN_SALT_LENGTH = 16
MAX_SALT_LENGTH = 64
OPENSSL_MEMORY_LIMIT = (2**31) - 1


@dataclass(frozen=True, slots=True)
class SCryptSettings:
    """
    Settings for SCrypt Password Hashing

    Args:
        - salt_length: Size of the salt in bytes
        - ln: Compute Cost factor (Every +1 causes CPU/Memory to Double)
        - r: Memory Block Size
        - p: Parallelism factor
        - dklen: Size of Digest in bytes
        - max_password_length: Length of password string
    """

    ln: int
    r: int
    p: int
    dklen: int
    salt_length: int
    max_password_length: int

    @property
    def maxmem(self) -> int:
        return 128 * self.r * ((1 << self.ln) + self.p + 2) + (1 << 20)

    def __post_init__(self):
        if self.ln <= 0 or self.r <= 0 or self.p <= 0:
            raise InvalidSettingsValue("Any of ln, r, p is 0")
        if not (MIN_SALT_LENGTH <= self.salt_length <= MAX_SALT_LENGTH):
            raise InvalidSettingsValue(
                f"Salt Range out of Range. Recommended: [{MIN_SALT_LENGTH}, {MAX_SALT_LENGTH}]"
            )
        if self.maxmem > OPENSSL_MEMORY_LIMIT:
            raise InvalidSettingsValue(
                "Possibility of Out-of-Memory. Reduce the values of ln, r, p"
            )


DEFAULT_SETTINGS: SCryptSettings = SCryptSettings(
    salt_length=32, ln=17, r=8, p=1, dklen=32, max_password_length=256
)
