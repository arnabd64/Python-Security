"""
Module: python_security.scrypt.core

Description:
    - Public functions available for Password Hashing
    - `hash_password()` to hash a plain text password
    - `verify_password()` to verify a plain text password with a Digest
"""

import hmac
import secrets

from python_security.scrypt import utils
from python_security.scrypt.config import SCryptSettings


def hash_password(
    plain_text_password: str, *, settings: SCryptSettings = SCryptSettings.default()
) -> str:
    """
    Hashes the plain_text_password with respect to the settings
    and returns a Password Hashing Competition compliant string.

    Ex: $scrypt$ln=17,r=8,p=1$salt$hash

    Args:
        - plain_text_password
        - settings

    Retruns:
        - str
    """
    # generate salt
    salt: bytes = secrets.token_bytes(settings.salt_length)

    # hash the password
    digest: bytes = utils.digest(plain_text_password, salt, settings)

    # convert to phc string
    return utils.phc_encoder(salt, digest, settings)


def verify_password(plain_text_password: str, hashed_password: str) -> bool:
    """
    Matches the plain_text_password with hashed_password

    Args:
        - plain_text_password
        - hashed_password

    Returns:
        - bool
    """
    # parse hashed password
    salt, actual_digest, settings = utils.phc_decoder(hashed_password)

    # hash plain_text_password
    recieved_digest: bytes = utils.digest(plain_text_password, salt, settings)

    return hmac.compare_digest(actual_digest, recieved_digest)
