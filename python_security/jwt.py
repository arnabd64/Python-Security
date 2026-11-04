import base64
import hashlib
import hmac
import json
import time
import uuid


class JWT:
    MIN_SECRET_LENGTH: int = 16
    MAX_SECRET_LENGTH: int = 64

    def __init__(
        self,
        secret: str,
        access_ttl: int = 900,
        refresh_ttl: int = 432000,
        iss: str = "JWTService",
        aud: str = "All",
    ) -> None:
        # validation
        if not (self.MIN_SECRET_LENGTH <= len(secret) <= self.MAX_SECRET_LENGTH):
            raise ValueError(
                f"Length of Secret is out of range ({self.MIN_SECRET_LENGTH}, {self.MAX_SECRET_LENGTH})"
            )
        if refresh_ttl < access_ttl:
            raise ValueError("Refresh TTL is shorter than Access TTL")

        try:
            jti_generator = uuid.uuid7
        except Exception:
            jti_generator = uuid.uuid4

        # assignment
        self.secret = secret
        self.access_ttl = access_ttl
        self.refresh_ttl = refresh_ttl
        self.iss = iss
        self.aud = aud
        self._jti_generator = jti_generator
        self._JWT_HEADER = {"alg": "HS256", "typ": "JWT"}

    def _payload_encoder(self, payload: dict) -> bytes:
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    def _payload_decoder(self, payload: bytes) -> dict:
        return json.loads(payload)

    def _b64_encoder(self, data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")

    def _b64_decoder(self, data: str) -> bytes:
        return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))

    def _sign(self, b64_header: str, b64_payload: str) -> bytes:
        body: str = f"{b64_header}.{b64_payload}"
        return hmac.new(
            key=self.secret.encode("utf-8"),
            msg=body.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()

    def _verify_signature(
        self, b64_header: str, b64_payload: str, b64_signature: str
    ) -> bool:
        return hmac.compare_digest(
            self._b64_decoder(b64_signature), self._sign(b64_header, b64_payload)
        )

    def _token(self, payload: dict) -> str:
        # encode parts
        b64_header: str = self._b64_encoder(self._payload_encoder(self._JWT_HEADER))
        b64_payload: str = self._b64_encoder(self._payload_encoder(payload))

        # sign the header & payload
        b64_signature: str = self._b64_encoder(self._sign(b64_header, b64_payload))

        return f"{b64_header}.{b64_payload}.{b64_signature}"

    def issue_tokens(self, payload: dict) -> tuple[str, ...]:
        # timestamp
        now: int = int(time.time())
        access_jti: str = self._jti_generator().hex
        refresh_jti: str = self._jti_generator().hex

        # payload for Access Token
        access_payload = {
            **payload,
            "iss": self.iss,
            "aud": self.aud,
            "jti": access_jti,
            "iat": now,
            "exp": now + self.access_ttl,
            "token_type": "access",
        }

        # payload for Refresh Token
        refresh_payload = {
            **payload,
            "iss": self.iss,
            "aud": self.aud,
            "jti": refresh_jti,
            "iat": now,
            "exp": now + self.refresh_ttl,
            "token_type": "refresh",
        }

        # generate tokens
        access: str = self._token(access_payload)
        refresh: str = self._token(refresh_payload)

        return access, refresh, access_jti, refresh_jti

    def decode_token(self, token: str) -> dict:
        # validation
        parts: list[str] = token.split(".")
        if len(parts) != 3:
            raise ValueError("Token should have 3 parts separated by '.'")

        # extract parts
        b64_header, b64_payload, b64_signature = parts

        # validate sign
        if not self._verify_signature(*parts):
            raise ValueError("Invalid Signature")

        # validate header
        header = self._payload_decoder(self._b64_decoder(b64_header))
        if header["alg"] != self._JWT_HEADER["alg"]:
            raise ValueError("Invalid JWT Sign Algorithm")

        # extract payload
        payload: dict = self._payload_decoder(self._b64_decoder(b64_payload))

        # Verify Structure using dataclasses
        # Not Implemented

        # check expiry
        now: int = int(time.time())
        if payload["exp"] < now:
            raise ValueError("Token has Expired")

        # validate issuer and audience
        if payload.get("iss") != self.iss:
            raise ValueError("Invalid Token Issuer")
        if payload.get("aud") != self.aud:
            raise ValueError("Invalid Token Audience")

        return payload

    def verify_token(self, token: str) -> bool:
        try:
            _ = self.decode_token(token)
            return True
        except ValueError:
            return False

    _RESERVED_CLAIMS: frozenset[str] = frozenset(
        {"iss", "aud", "jti", "iat", "exp", "token_type"}
    )

    def renew_access_token(self, refresh_token: str) -> str:
        refresh_payload: dict = self.decode_token(refresh_token)
        if not refresh_payload["token_type"] == "refresh":
            raise ValueError("Expected Refresh token")

        # strip reserved claims so issue_tokens can re-stamp them fresh
        user_payload: dict = {
            k: v for k, v in refresh_payload.items() if k not in self._RESERVED_CLAIMS
        }

        access, _refresh, _access_jti, _refresh_jti = self.issue_tokens(user_payload)
        return access
