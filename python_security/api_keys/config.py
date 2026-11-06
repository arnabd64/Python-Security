"""
Module: python_security.api_keys.config

Description:
    - Settings for API Keys
"""

from dataclasses import dataclass

from python_security.exceptions import InvalidSettings

MAX_API_KEY_LENGTH: int = 64


@dataclass(frozen=True, slots=True)
class StatelessAPIKeySettings:
    length: int
    prefix: str

    def __post_init__(self) -> None:
        if not (8 <= self.length <= MAX_API_KEY_LENGTH):
            raise InvalidSettings(f"length should be within [8, {MAX_API_KEY_LENGTH}]")
