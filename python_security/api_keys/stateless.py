"""
Module: python_security.api_keys.stateless

Description:
    - Public Facing APIs for generating Stateless API keys
    - Quickest way to generate key
"""

import secrets

from python_security.api_keys import utils
from python_security.api_keys.config import StatelessAPIKeySettings
from python_security.exceptions import InvalidPrefix

DEFAULT_SETTINGS = StatelessAPIKeySettings.default()


def new_stateless_api_key(
    settings: StatelessAPIKeySettings = DEFAULT_SETTINGS,
) -> tuple[str, str]:
    """
    Generates an API Key where the actual key secret: 'sk-{key}' is a random
    cryptographically generated string. Good for simple use cases.

    Args:
        - `settings` (StatelessAPIKeySettings): Configuration for the key

    Returns:
        - `api_key` (str): The API Key
        - `key_hash` (str): Hex key hash that can be used to verify
    """
    # generate the key
    key: str = secrets.token_hex(settings.length)

    # generate a hash for the key
    digest: str = utils.key_digest(key).hex()

    return f"{settings.prefix}-{key}", digest


def verify_stateless_api_key(
    key: str,
    digest: str,
    settings: StatelessAPIKeySettings = DEFAULT_SETTINGS,
) -> bool:
    """
    Verifies a stateless API Key

    Args:
        - `key` (str) : Submitted API Key
        - `digest` (str) : Expected hash for the API Key
        - `expected_prefix` (str) : Prefix to check
    """
    # validate if key, digest
    if not isinstance(key, str):
        raise TypeError("key is not str")
    if not isinstance(digest, str):
        raise TypeError("digest is not str")
    if not isinstance(settings, StatelessAPIKeySettings):
        raise TypeError("invalid settings type")

    # check if key has the prefix
    if not key.startswith(settings.prefix):
        raise InvalidPrefix

    # extract the key
    _, secret = key.split("-")

    # verify the secret
    return utils.verify_digest(secret, digest)
