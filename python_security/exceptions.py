"""
Module: python_security.exceptions

Description:
    - Module Level Exceptions
"""


# --------- Common Exceptions ------
class InvalidSettings(Exception): ...


# --------- SCrypt Exceptions ------


class SCryptException(Exception): ...


class EncodeError(SCryptException): ...


class DecodeError(SCryptException): ...


# --------- JWT Exceptions ---------
class JWTException(Exception): ...


class TokenDecodeError(JWTException): ...


class PayloadError(JWTException): ...


class SignatureError(JWTException): ...


class ExpiredTokenError(JWTException): ...


# -------- API Keys ----------------
class APIKeyException(Exception): ...


class InvalidPrefix(APIKeyException): ...
