"""
Module: python_security.scrypt.exceptions

Description:
    - custom exceptions
"""


# Base Exception
class SCryptException(Exception): ...


class InvalidSettingsValue(SCryptException): ...


class EncodeError(SCryptException): ...


class DecodeError(SCryptException): ...
