"""
Module: python_security.scrypt.utils

Description:
    - Utility & Helper function for core module
"""

import base64
import binascii
import hashlib

from python_security.scrypt.config import SCryptSettings
from python_security.scrypt.exceptions import DecodeError, EncodeError


def b64_encoder(data: bytes) -> str:
    return base64.b64encode(data).rstrip(b"=").decode("ascii")


def b64_decoder(data: str) -> bytes:
    try:
        return base64.b64decode(data + "=" * (-len(data) % 4), validate=True)
    except (binascii.Error, ValueError) as err:
        raise DecodeError(f"Invalid base64 data: {err}") from err


def phc_encoder(salt: bytes, digest: bytes, settings: SCryptSettings) -> str:
    return f"$scrypt$ln={settings.ln},r={settings.r},p={settings.p}${b64_encoder(salt)}${b64_encoder(digest)}"


def phc_decoder(hashed_password: str):
    if not isinstance(hashed_password, str):
        raise DecodeError("hashed_password is not a string")

    # extract the parts
    parts = hashed_password.split("$")
    if len(parts) != 5:
        raise DecodeError("hashed_password is malformed")

    empty, algo, params, b64_salt, b64_digest = parts

    if empty or algo != "scrypt":
        raise DecodeError("hashed_password is malformed")

    # extract components
    kv: dict[str, int] = {}
    for param in params.split(","):
        key, value = param.split("=")
        kv[key] = int(value)

    salt: bytes = b64_decoder(b64_salt)
    digest: bytes = b64_decoder(b64_digest)
    settings: SCryptSettings = SCryptSettings(
        ln=kv["ln"],
        r=kv["r"],
        p=kv["p"],
        dklen=len(digest),
        salt_length=len(salt),
        max_password_length=len(hashed_password),  # Not needed but mandatory parameter
    )

    return salt, digest, settings


def digest(plain_text_password: str, salt: bytes, settings: SCryptSettings) -> bytes:
    if len(plain_text_password) > settings.max_password_length:
        raise EncodeError(
            f"Password is Too Long. Max Allowed Chars: {settings.max_password_length}"
        )
    return hashlib.scrypt(
        plain_text_password.encode(),
        salt=salt,
        n=(1 << settings.ln),
        r=settings.r,
        p=settings.p,
        dklen=settings.dklen,
        maxmem=settings.maxmem,
    )
