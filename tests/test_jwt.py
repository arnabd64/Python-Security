"""
Security-focused test suite for ``python_security.jwt``.

Tooling : pytest + hypothesis + pytest-cov
Run     : pytest test_jwt_security.py \
              --cov=python_security.jwt --cov-branch \
              --cov-report=term-missing --cov-fail-under=98

Design notes
------------
* Every test asserts the behaviour a *production-grade* JWT library must have.
  Tests that expose a real flaw in the current implementation FAIL normally
  (no xfail) so that CI flags the flaw until it is fixed.  Each such test is
  marked with ``@pytest.mark.vuln`` (registered below) so they can be selected
  with ``-m vuln`` or excluded with ``-m "not vuln"``.
* Tokens used for attack simulation are crafted *independently* of the module
  under test (own base64url / HMAC helpers) so the tests do not share bugs
  with the code they check.
* Only ``python_security.jwt.core.time`` is frozen when time matters, so
  hypothesis internals are not affected.

Threat coverage (OWASP JWT cheat sheet / RFC 8725 / RFC 7519 / RFC 7515)
------------------------------------------------------------------------
alg=none, alg confusion, header tampering, ``crit``/``typ`` handling, signature
stripping and tampering, signature malleability (non-canonical base64url,
padding, junk characters, whitespace), wrong secret, key length (RFC 7518 3.2),
weak / missing / leaked secrets, expiry boundary, ``iat``/``nbf`` validation,
iss/aud confusion, access<->refresh token confusion, refresh-token privilege
escalation, session-lifetime extension, claim injection / reserved-claim
shadowing, type confusion in payload fields, malformed / hostile input
(fuzzing, giant tokens, deep nesting, non-UTF8, NaN/Infinity), exception
hygiene (no unexpected exception types, no secret leakage), constant-time
comparison, jti uniqueness (including under threads) and API surface.
"""

import base64
import hashlib
import hmac
import importlib.util
import json
import re
import secrets
import string
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from unittest import mock

import pytest
from hypothesis import HealthCheck, assume, given
from hypothesis import settings as hy_settings
from hypothesis import strategies as st

from python_security import jwt as jwt_pkg
from python_security.exceptions import (
    ExpiredTokenError,
    InvalidSettings,
    JWTException,
    PayloadError,
    SignatureError,
    TokenDecodeError,
)
from python_security.jwt import config, core, utils
from python_security.jwt.config import JWTSettings
from python_security.jwt.payload import Payload

hy_settings.register_profile(
    "jwt-security",
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
hy_settings.load_profile("jwt-security")

vuln = pytest.mark.vuln  # tests that currently expose a real weakness

# --------------------------------------------------------------------------- #
# Helpers (independent re-implementation of the wire format)
# --------------------------------------------------------------------------- #

SECRET = secrets.token_hex(32)  # 64 chars -> valid under any sane length policy
OTHER_SECRET = secrets.token_hex(32)
B64_ALPHABET = string.ascii_letters + string.digits + "-_"
HEADER = {"alg": "HS256", "typ": "JWT"}
ENV_KEYS = (
    "JWT_SECRET",
    "JWT_ACCESS_TTL",
    "JWT_REFRESH_TTL",
    "JWT_ISSUER",
    "JWT_AUDIENCE",
)
_DROP = object()


def make_settings(**over) -> JWTSettings:
    base = dict(
        secret=SECRET,
        access_ttl=900,
        refresh_ttl=86400,
        iss="test-issuer",
        aud="test-audience",
    )
    base.update(over)
    return JWTSettings(**base)


S = make_settings()


def b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def jdump(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def hmac_b64(h: str, p: str, secret: str = SECRET) -> str:
    return b64(hmac.new(secret.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest())


def claims_dict(token_type: str = "access", **over) -> dict:
    now = int(time.time())
    data = {
        "jti": uuid.uuid4().hex,
        "sub": "user-1",
        "iat": now,
        "exp": now + 900,
        "iss": S.iss,
        "aud": S.aud,
        "claims": {"role": "user"},
        "token_type": token_type,
    }
    for key, value in over.items():
        if value is _DROP:
            data.pop(key, None)
        else:
            data[key] = value
    return data


def craft(
    payload=None,
    *,
    header=None,
    raw_header: bytes | None = None,
    raw_payload: bytes | None = None,
    secret: str = SECRET,
    signature: str | None = None,
) -> str:
    """Build a token without using the code under test."""
    h = b64(
        raw_header
        if raw_header is not None
        else jdump(HEADER if header is None else header)
    )
    p = b64(
        raw_payload
        if raw_payload is not None
        else jdump(claims_dict() if payload is None else payload)
    )
    sig = signature if signature is not None else hmac_b64(h, p, secret)
    return f"{h}.{p}.{sig}"


def decode_segment(segment: str):
    return json.loads(base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4)))


def flip_last_char_padding_bit(sig: str) -> str:
    """Change only the unused trailing bits of the final base64url char."""
    return sig[:-1] + B64_ALPHABET[B64_ALPHABET.index(sig[-1]) ^ 1]


@contextmanager
def frozen(now: int):
    """Freeze the clock seen by ``core`` only."""
    with mock.patch.object(core, "time") as fake:
        fake.time.return_value = float(now)
        yield


def load_config(monkeypatch, **env):
    """Execute config.py afresh under a controlled environment."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    name = "_jwt_config_under_test"
    spec = importlib.util.spec_from_file_location(name, config.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


claims_st = st.dictionaries(
    st.text(max_size=20),
    st.one_of(
        st.text(max_size=50),
        st.integers(min_value=-(2**53), max_value=2**53),
        st.floats(allow_nan=False, allow_infinity=False),
    ),
    max_size=8,
)
sub_st = st.text(min_size=1, max_size=64)
secret_st = st.binary(min_size=32, max_size=32).map(bytes.hex)


# =========================================================================== #
# 1. config.JWTSettings
# =========================================================================== #
class TestSettings:
    def test_valid_settings_are_accepted(self):
        s = make_settings()
        assert (s.secret, s.access_ttl, s.refresh_ttl) == (SECRET, 900, 86400)

    def test_header_declares_hs256_jwt(self):
        assert S.header == {"alg": "HS256", "typ": "JWT"}

    def test_header_returns_a_fresh_dict_each_time(self):
        first = S.header
        first["alg"] = "none"
        assert S.header["alg"] == "HS256"

    @pytest.mark.parametrize(
        "field", ["secret", "access_ttl", "refresh_ttl", "iss", "aud"]
    )
    def test_settings_are_immutable(self, field):
        with pytest.raises(FrozenInstanceError):
            setattr(S, field, "x")

    def test_settings_use_slots(self):
        assert not hasattr(S, "__dict__")

    @pytest.mark.parametrize("length", [0, 1, 8, 15])
    def test_secret_below_minimum_is_rejected(self, length):
        with pytest.raises(InvalidSettings):
            make_settings(secret="k" * length)

    @pytest.mark.parametrize("length", [65, 128, 1024])
    def test_secret_above_maximum_is_rejected(self, length):
        with pytest.raises(InvalidSettings):
            make_settings(secret=secrets.token_hex(length)[:length])

    @pytest.mark.parametrize("length", [32, 48, 64])
    def test_secret_in_valid_range_is_accepted(self, length):
        assert make_settings(secret=secrets.token_hex(length)[:length]).secret

    @vuln
    @pytest.mark.parametrize("length", [16, 20, 24, 31])
    def test_secret_shorter_than_256_bits_is_rejected(self, length):
        """RFC 7518 §3.2: HS256 keys MUST be at least 256 bits (32 bytes)."""
        with pytest.raises(InvalidSettings):
            make_settings(secret=secrets.token_hex(length)[:length])

    def test_refresh_ttl_shorter_than_access_ttl_is_rejected(self):
        with pytest.raises(InvalidSettings):
            make_settings(access_ttl=1000, refresh_ttl=999)

    def test_refresh_ttl_longer_than_access_ttl_is_accepted(self):
        assert make_settings(access_ttl=60, refresh_ttl=61).refresh_ttl == 61

    @vuln
    @pytest.mark.parametrize("ttl", [0, -1, -900])
    def test_non_positive_access_ttl_is_rejected(self, ttl):
        with pytest.raises(InvalidSettings):
            make_settings(access_ttl=ttl, refresh_ttl=86400)

    @vuln
    @pytest.mark.parametrize(
        "bad",
        [
            {"access_ttl": "900"},
            {"access_ttl": 900.5},
            {"access_ttl": None},
            {"access_ttl": True},
            {"refresh_ttl": "86400"},
            {"refresh_ttl": 86400.5},
            {"refresh_ttl": None},
        ],
        ids=str,
    )
    def test_non_integer_ttls_are_rejected(self, bad):
        with pytest.raises(InvalidSettings):
            make_settings(**bad)

    @vuln
    @pytest.mark.parametrize("bad", [{"iss": ""}, {"aud": ""}], ids=str)
    def test_empty_issuer_or_audience_is_rejected(self, bad):
        with pytest.raises(InvalidSettings):
            make_settings(**bad)

    @vuln
    @pytest.mark.parametrize(
        "secret",
        [b"x" * 32, 12345, None, ["a"] * 32],
        ids=["bytes", "int", "none", "list"],
    )
    def test_non_string_secret_is_rejected(self, secret):
        with pytest.raises(InvalidSettings):
            make_settings(secret=secret)

    @vuln
    @pytest.mark.parametrize(
        "secret", ["a" * 32, "0" * 32, "ab" * 16, "password" * 4, "1234567890" * 4]
    )
    def test_low_entropy_secret_is_rejected(self, secret):
        with pytest.raises(InvalidSettings):
            make_settings(secret=secret)

    @vuln
    def test_repr_does_not_leak_secret(self):
        assert SECRET not in repr(S)

    @vuln
    def test_str_and_format_do_not_leak_secret(self):
        assert SECRET not in str(S)
        assert SECRET not in f"{S}"
        assert SECRET not in f"{S!r}"

    def test_validation_error_does_not_echo_the_secret(self):
        with pytest.raises(InvalidSettings) as exc:
            make_settings(secret="abcde")
        assert "abcde" not in str(exc.value)

    @vuln
    @pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
    def test_missing_secret_in_environment_fails_closed(self, monkeypatch, value):
        """A random per-process fallback silently breaks multi-worker deployments
        and invalidates all tokens on restart; production must refuse to start."""
        env = {} if value is None else {"JWT_SECRET": value}
        with pytest.raises(InvalidSettings):
            load_config(monkeypatch, **env)

    def test_secret_is_read_from_environment(self, monkeypatch):
        mod = load_config(monkeypatch, JWT_SECRET=SECRET)
        assert mod.DEFAULT_SETTINGS.secret == SECRET

    def test_ttls_are_read_from_environment(self, monkeypatch):
        mod = load_config(
            monkeypatch, JWT_SECRET=SECRET, JWT_ACCESS_TTL="60", JWT_REFRESH_TTL="3600"
        )
        assert (mod.DEFAULT_SETTINGS.access_ttl, mod.DEFAULT_SETTINGS.refresh_ttl) == (
            60,
            3600,
        )

    def test_ttl_defaults_are_short_lived_access_long_lived_refresh(self, monkeypatch):
        mod = load_config(monkeypatch, JWT_SECRET=SECRET)
        assert mod.DEFAULT_SETTINGS.access_ttl == 15 * 60
        assert mod.DEFAULT_SETTINGS.refresh_ttl == 5 * 24 * 3600

    def test_issuer_and_audience_come_from_environment(self, monkeypatch):
        mod = load_config(
            monkeypatch, JWT_SECRET=SECRET, JWT_ISSUER="me", JWT_AUDIENCE="you"
        )
        assert (mod.DEFAULT_SETTINGS.iss, mod.DEFAULT_SETTINGS.aud) == ("me", "you")

    def test_issuer_and_audience_defaults(self, monkeypatch):
        mod = load_config(monkeypatch, JWT_SECRET=SECRET)
        assert (mod.DEFAULT_SETTINGS.iss, mod.DEFAULT_SETTINGS.aud) == (
            "JWTService",
            "All",
        )

    @pytest.mark.parametrize("raw", ["abc", "1.5"])
    def test_non_numeric_ttl_in_environment_is_rejected(self, monkeypatch, raw):
        with pytest.raises((InvalidSettings, ValueError)):
            load_config(monkeypatch, JWT_SECRET=SECRET, JWT_ACCESS_TTL=raw)

    @vuln
    def test_zero_ttl_in_environment_is_rejected(self, monkeypatch):
        with pytest.raises(InvalidSettings):
            load_config(monkeypatch, JWT_SECRET=SECRET, JWT_ACCESS_TTL="0")

    def test_refresh_shorter_than_access_in_environment_is_rejected(self, monkeypatch):
        with pytest.raises(InvalidSettings):
            load_config(
                monkeypatch,
                JWT_SECRET=SECRET,
                JWT_ACCESS_TTL="1000",
                JWT_REFRESH_TTL="10",
            )

    def test_module_level_default_settings_are_valid(self):
        d = config.DEFAULT_SETTINGS
        assert isinstance(d, JWTSettings)
        assert config.MIN_SECRET_LENGTH <= len(d.secret) <= config.MAX_SECRET_LENGTH
        assert d.refresh_ttl >= d.access_ttl


# =========================================================================== #
# 2. payload.Payload
# =========================================================================== #
def _payload(**over) -> Payload:
    data = dict(
        jti="a" * 32,
        sub="u",
        iat=100,
        exp=200,
        iss="i",
        aud="a",
        claims={"k": "v"},
        token_type="access",
    )
    data.update(over)
    return Payload(**data)


class TestPayload:
    def test_valid_payload(self):
        p = _payload()
        assert (p.sub, p.token_type, p.claims) == ("u", "access", {"k": "v"})

    def test_to_dict_has_exactly_the_expected_keys(self):
        assert set(_payload().to_dict()) == {
            "jti",
            "sub",
            "iat",
            "exp",
            "iss",
            "aud",
            "claims",
            "token_type",
        }

    def test_dict_roundtrip(self):
        p = _payload()
        assert Payload.from_dict(p.to_dict()) == p

    def test_claims_may_be_none(self):
        assert Payload.from_dict(_payload(claims=None).to_dict()).claims is None

    def test_payload_is_immutable(self):
        with pytest.raises(FrozenInstanceError):
            _payload().sub = "admin"

    @pytest.mark.parametrize("bad", ["ACCESS", "Access", "admin", "", None, 1])
    def test_invalid_token_type_is_rejected(self, bad):
        with pytest.raises(PayloadError):
            _payload(token_type=bad)

    def test_exp_before_iat_is_rejected(self):
        with pytest.raises(PayloadError):
            _payload(iat=200, exp=199)

    @vuln
    @pytest.mark.parametrize(
        "key", ["jti", "sub", "iat", "exp", "iss", "aud", "claims", "token_type"]
    )
    def test_from_dict_missing_key_raises_payload_exception(self, key):
        data = _payload().to_dict()
        del data[key]
        with pytest.raises(PayloadError):
            Payload.from_dict(data)

    @vuln
    @pytest.mark.parametrize("bad", [None, [], "x", 5], ids=repr)
    def test_from_dict_non_mapping_raises_payload_exception(self, bad):
        with pytest.raises(PayloadError):
            Payload.from_dict(bad)

    @vuln
    @pytest.mark.parametrize(
        "field,value",
        [
            ("sub", 123),
            ("sub", None),
            ("iss", ["x"]),
            ("aud", None),
            ("iat", "123"),
            ("iat", "abc"),
            ("iat", True),
            ("iat", None),
            ("exp", "999"),
            ("exp", [1]),
            ("exp", True),
            ("jti", None),
            ("jti", 123),
            ("claims", "x"),
            ("claims", [1]),
        ],
        ids=lambda v: repr(v)[:20],
    )
    def test_from_dict_rejects_wrongly_typed_fields(self, field, value):
        data = _payload().to_dict()
        data[field] = value
        with pytest.raises(PayloadError):
            Payload.from_dict(data)

    def test_from_dict_ignores_unknown_keys(self):
        data = _payload().to_dict()
        data.update({"admin": True, "is_superuser": 1})
        p = Payload.from_dict(data)
        assert not hasattr(p, "admin") and not hasattr(p, "is_superuser")

    @pytest.mark.parametrize(
        "exc", [TokenDecodeError, ExpiredTokenError, SignatureError, PayloadError]
    )
    def test_all_library_errors_share_the_jwt_exception_base(self, exc):
        """Callers must be able to catch one base class (verify_token relies on it)."""
        assert issubclass(exc, JWTException)


# =========================================================================== #
# 3. utils
# =========================================================================== #
class TestUtils:
    # ---- JSON ----------------------------------------------------------- #
    def test_json_encode_is_compact_utf8_bytes(self):
        out = utils.json_encode({"a": 1, "b": [1, 2]})
        assert isinstance(out, bytes) and out == b'{"a":1,"b":[1,2]}'

    def test_json_encode_escapes_non_ascii_deterministically(self):
        assert utils.json_encode({"k": "é"}) == b'{"k":"\\u00e9"}'

    def test_json_decode_returns_dict(self):
        assert utils.json_decode(b'{"a":1}') == {"a": 1}

    @vuln
    @pytest.mark.parametrize(
        "value", [float("nan"), float("inf"), float("-inf")], ids=repr
    )
    def test_json_encode_rejects_non_finite_numbers(self, value):
        with pytest.raises((ValueError, JWTException)):
            utils.json_encode({"x": value})

    @vuln
    @pytest.mark.parametrize(
        "raw",
        [b'{"a": NaN}', b'{"a": Infinity}', b'{"a": -Infinity}'],
        ids=bytes.decode,
    )
    def test_json_decode_rejects_non_finite_numbers(self, raw):
        with pytest.raises((ValueError, JWTException)):
            utils.json_decode(raw)

    # ---- base64url ------------------------------------------------------ #
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (b"", ""),
            (b"f", "Zg"),
            (b"fo", "Zm8"),
            (b"foo", "Zm9v"),
            (b"foob", "Zm9vYg"),
            (b"fooba", "Zm9vYmE"),
            (b"foobar", "Zm9vYmFy"),
        ],
    )
    def test_b64_encode_rfc4648_vectors_without_padding(self, raw, expected):
        assert utils.b64_encode(raw) == expected

    def test_b64_encode_uses_url_safe_alphabet(self):
        assert utils.b64_encode(b"\xfb\xff\xfe") == "-__-"

    @pytest.mark.parametrize("text", ["Zg", "Zg==", "Zm8", "Zm8=", "Zm9v"])
    def test_b64_decode_accepts_padded_and_unpadded(self, text):
        assert utils.b64_decode(text) == base64.urlsafe_b64decode(
            text + "=" * (-len(text) % 4)
        )

    def test_b64_decode_rejects_impossible_length(self):
        with pytest.raises(ValueError):
            utils.b64_decode("a")

    @vuln
    @pytest.mark.parametrize(
        "bad", ["ab!d", "ab d", "ab+d", "ab/d", "ab\nd", "ab\x00d", "éééé"], ids=repr
    )
    def test_b64_decode_is_strict_about_alphabet(self, bad):
        """Non-alphabet characters must not be silently discarded/translated."""
        with pytest.raises((ValueError, JWTException)):
            utils.b64_decode(bad)

    @vuln
    def test_b64_decode_rejects_non_canonical_trailing_bits(self):
        # "Zg" is canonical for b"f"; "Zh" decodes to the same byte with dirty bits
        assert utils.b64_decode("Zg") == b"f"
        with pytest.raises((ValueError, JWTException)):
            utils.b64_decode("Zh")

    # ---- sign / verify_signature --------------------------------------- #
    def test_sign_matches_independent_hmac_sha256(self):
        expected = hmac.new(SECRET.encode(), b"h.p", hashlib.sha256).digest()
        assert utils.sign("h", "p", SECRET) == expected

    def test_sign_is_deterministic_and_32_bytes(self):
        assert utils.sign("h", "p", SECRET) == utils.sign("h", "p", SECRET)
        assert len(utils.sign("h", "p", SECRET)) == 32

    def test_sign_depends_on_secret_header_and_payload(self):
        base = utils.sign("h", "p", SECRET)
        assert utils.sign("h", "p", OTHER_SECRET) != base
        assert utils.sign("H", "p", SECRET) != base
        assert utils.sign("h", "P", SECRET) != base

    def test_sign_encodes_non_ascii_secret_as_utf8(self):
        secret = "sécrèt-" + "x" * 30
        expected = hmac.new(secret.encode("utf-8"), b"h.p", hashlib.sha256).digest()
        assert utils.sign("h", "p", secret) == expected

    def _parts(self):
        h, p = b64(jdump(HEADER)), b64(jdump(claims_dict()))
        return h, p, hmac_b64(h, p)

    def test_verify_signature_accepts_valid(self):
        h, p, s = self._parts()
        assert utils.verify_signature(h, p, s, SECRET) is True

    def test_verify_signature_rejects_wrong_secret(self):
        h, p, s = self._parts()
        assert utils.verify_signature(h, p, s, OTHER_SECRET) is False

    def test_verify_signature_rejects_tampered_payload_and_header(self):
        h, p, s = self._parts()
        assert utils.verify_signature(h, p + "A", s, SECRET) is False
        assert utils.verify_signature("A" + h, p, s, SECRET) is False

    @pytest.mark.parametrize(
        "mutate",
        [lambda s: "", lambda s: s[:-4], lambda s: s + "AAAA"],
        ids=["empty", "truncated", "extended"],
    )
    def test_verify_signature_rejects_wrong_length_signatures(self, mutate):
        h, p, s = self._parts()
        assert utils.verify_signature(h, p, mutate(s), SECRET) is False

    def test_verify_signature_uses_constant_time_compare(self):
        h, p, s = self._parts()
        with mock.patch.object(
            utils.hmac, "compare_digest", wraps=hmac.compare_digest
        ) as spy:
            utils.verify_signature(h, p, s, SECRET)
        assert spy.call_count == 1
        assert all(isinstance(a, bytes) for a in spy.call_args.args)

    @vuln
    def test_verify_signature_rejects_non_canonical_encoding(self):
        h, p, s = self._parts()
        assert (
            utils.verify_signature(h, p, flip_last_char_padding_bit(s), SECRET) is False
        )

    @vuln
    @pytest.mark.parametrize("bad_sig", ["a", "é", "!", "a" * 44 + "!"], ids=repr)
    def test_verify_signature_never_raises_on_bad_encoding(self, bad_sig):
        h, p, _ = self._parts()
        assert utils.verify_signature(h, p, bad_sig, SECRET) is False

    # ---- jti ------------------------------------------------------------ #
    def test_jti_is_32_lowercase_hex_chars(self):
        assert re.fullmatch(r"[0-9a-f]{32}", utils.generate_jti())

    def test_jti_values_are_unique(self):
        assert len({utils.generate_jti() for _ in range(5000)}) == 5000

    def test_jti_prefers_uuid7_when_available(self, monkeypatch):
        fake = uuid.UUID(int=0x0123456789ABCDEF0123456789ABCDEF)
        monkeypatch.setattr(uuid, "uuid7", lambda: fake, raising=False)
        assert utils.generate_jti() == fake.hex

    def test_jti_falls_back_to_uuid4(self, monkeypatch):
        monkeypatch.delattr(uuid, "uuid7", raising=False)
        monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=7))
        assert utils.generate_jti() == uuid.UUID(int=7).hex

    # ---- issue_token / decode_token ------------------------------------- #
    def test_issue_token_structure_and_signature(self):
        p = Payload(**{**claims_dict(), "claims": {"a": 1}})
        token, jti = utils.issue_token(p, S)
        h, body, sig = token.split(".")
        assert jti == p.jti
        assert decode_segment(h) == HEADER
        assert decode_segment(body) == p.to_dict()
        assert sig == hmac_b64(h, body)

    def test_issue_token_uses_only_url_safe_characters(self):
        token, _ = utils.issue_token(Payload(**claims_dict()), S)
        assert re.fullmatch(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", token)

    def test_issue_token_does_not_embed_the_secret(self):
        token, _ = utils.issue_token(Payload(**claims_dict()), S)
        assert SECRET not in token
        assert b64(SECRET.encode()) not in token

    def test_decode_token_roundtrip(self):
        p = Payload(**claims_dict())
        token, _ = utils.issue_token(p, S)
        h, body, sig = token.split(".")
        header, payload, signature = utils.decode_token(h, body, sig)
        assert (
            header == HEADER
            and payload == p
            and signature == base64.urlsafe_b64decode(sig + "=")
        )

    @vuln
    @pytest.mark.parametrize("which", [0, 1, 2])
    def test_decode_token_rejects_garbage_with_library_exception(self, which):
        parts = [b64(jdump(HEADER)), b64(jdump(claims_dict())), "AAAA"]
        parts[which] = "!!!!a"
        with pytest.raises(JWTException):
            utils.decode_token(*parts)


# =========================================================================== #
# 4. core.issue_access_token / issue_refresh_token
# =========================================================================== #
NOW = 1_700_000_000


class TestIssue:
    def test_returns_token_and_jti_strings(self):
        token, jti = core.issue_access_token("u", {}, S)
        assert isinstance(token, str) and isinstance(jti, str)

    def test_access_token_lifetime_and_type(self):
        with frozen(NOW):
            token, jti = core.issue_access_token("u", {"a": 1}, S)
            p = core.parse_token(token, "access", S)
        assert (p.iat, p.exp, p.token_type, p.jti) == (NOW, NOW + 900, "access", jti)
        assert (p.iss, p.aud, p.sub, p.claims) == (S.iss, S.aud, "u", {"a": 1})

    def test_refresh_token_lifetime_and_type(self):
        with frozen(NOW):
            token, jti = core.issue_refresh_token("u", {"a": 1}, S)
            p = core.parse_token(token, "refresh", S)
        assert (p.iat, p.exp, p.token_type, p.jti) == (NOW, NOW + 86400, "refresh", jti)

    def test_refresh_outlives_access(self):
        with frozen(NOW):
            a = core.parse_token(core.issue_access_token("u", {}, S)[0], "access", S)
            r = core.parse_token(core.issue_refresh_token("u", {}, S)[0], "refresh", S)
        assert r.exp > a.exp

    def test_each_token_gets_a_unique_jti_and_signature(self):
        t1, j1 = core.issue_access_token("u", {}, S)
        t2, j2 = core.issue_access_token("u", {}, S)
        assert j1 != j2 and t1 != t2

    def test_unicode_subject_and_claims_roundtrip(self):
        token, _ = core.issue_access_token("üser-✓", {"nom": "Zoë", "emoji": "🔐"}, S)
        p = core.parse_token(token, "access", S)
        assert p.sub == "üser-✓" and p.claims == {"nom": "Zoë", "emoji": "🔐"}

    def test_empty_claims_are_supported(self):
        p = core.parse_token(core.issue_access_token("u", {}, S)[0], "access", S)
        assert p.claims == {}

    @pytest.mark.parametrize(
        "issuer", [core.issue_access_token, core.issue_refresh_token]
    )
    def test_reserved_claim_names_cannot_override_registered_fields(self, issuer):
        hostile = {
            "exp": 9_999_999_999,
            "sub": "admin",
            "iss": "evil",
            "aud": "evil",
            "token_type": "refresh",
            "jti": "forged",
        }
        token, jti = issuer("alice", hostile, S)
        kind = "access" if issuer is core.issue_access_token else "refresh"
        p = core.parse_token(token, kind, S)
        assert p.sub == "alice" and p.iss == S.iss and p.aud == S.aud
        assert p.token_type == kind and p.jti == jti and p.exp < 9_999_999_999
        assert p.claims == hostile  # kept inert, nested under "claims"

    def test_unserialisable_claims_are_rejected(self):
        with pytest.raises((TypeError, ValueError, JWTException)):
            core.issue_access_token("u", {"x": object()}, S)

    @vuln
    @pytest.mark.parametrize("value", [float("nan"), float("inf")], ids=repr)
    def test_non_finite_claims_are_rejected(self, value):
        """NaN/Infinity are not valid JSON; other parsers will misread the token."""
        with pytest.raises((ValueError, JWTException)):
            core.issue_access_token("u", {"x": value}, S)

    @vuln
    @pytest.mark.parametrize("sub", ["", None, 123], ids=repr)
    @pytest.mark.parametrize(
        "issuer", [core.issue_access_token, core.issue_refresh_token]
    )
    def test_invalid_subject_is_rejected(self, issuer, sub):
        with pytest.raises((JWTException, ValueError, TypeError)):
            issuer(sub, {}, S)

    def test_custom_settings_are_used_for_signing(self):
        token, _ = core.issue_access_token("u", {}, S)
        assert core.verify_token(token, "access", S) is True
        assert (
            core.verify_token(token, "access", make_settings(secret=OTHER_SECRET))
            is False
        )

    def test_default_settings_are_used_when_none_given(self):
        token, _ = core.issue_access_token("u", {})
        assert core.verify_token(token, "access") is True
        assert core.verify_token(token, "access", S) is False

    def test_tokens_issued_at_different_times_have_different_iat(self):
        with frozen(NOW):
            t1, _ = core.issue_access_token("u", {}, S)
            p1 = core.parse_token(t1, "access", S)
        with frozen(NOW + 5):
            t2, _ = core.issue_access_token("u", {}, S)
            p2 = core.parse_token(t2, "access", S)
        assert p2.iat - p1.iat == 5

    def test_issued_header_pins_algorithm(self):
        token, _ = core.issue_access_token("u", {}, S)
        assert decode_segment(token.split(".")[0]) == {"alg": "HS256", "typ": "JWT"}

    def test_issued_signature_matches_independent_hmac(self):
        token, _ = core.issue_access_token("u", {"k": "v"}, S)
        h, p, sig = token.split(".")
        assert sig == hmac_b64(h, p)

    def test_token_does_not_contain_the_secret(self):
        token, _ = core.issue_refresh_token("u", {}, S)
        assert SECRET not in token

    def test_jtis_are_unique_across_threads(self):
        def work(_):
            return [core.issue_access_token("u", {}, S)[1] for _ in range(250)]

        with ThreadPoolExecutor(max_workers=8) as pool:
            jtis = [j for batch in pool.map(work, range(8)) for j in batch]
        assert len(set(jtis)) == len(jtis) == 2000


# =========================================================================== #
# 5. core.parse_token
# =========================================================================== #
def _tamper_payload(token: str, **changes) -> str:
    h, p, s = token.split(".")
    body = decode_segment(p)
    body.update(changes)
    return f"{h}.{b64(jdump(body))}.{s}"  # old signature kept


class TestParseToken:
    # ---- happy path ----------------------------------------------------- #
    def test_valid_access_token(self):
        token, jti = core.issue_access_token("u", {"r": "x"}, S)
        p = core.parse_token(token, "access", S)
        assert isinstance(p, Payload) and p.jti == jti and p.claims == {"r": "x"}

    def test_valid_refresh_token(self):
        token, _ = core.issue_refresh_token("u", {}, S)
        assert core.parse_token(token, "refresh", S).token_type == "refresh"

    def test_default_expected_type_is_access(self):
        token, _ = core.issue_access_token("u", {}, S)
        assert core.parse_token(token, settings=S).token_type == "access"

    def test_independently_crafted_token_is_accepted(self):
        assert core.parse_token(craft(), "access", S).sub == "user-1"

    # ---- token type confusion ------------------------------------------ #
    def test_refresh_token_cannot_be_used_as_access_token(self):
        token, _ = core.issue_refresh_token("u", {}, S)
        with pytest.raises(JWTException):
            core.parse_token(token, "access", S)

    def test_access_token_cannot_be_used_as_refresh_token(self):
        token, _ = core.issue_access_token("u", {}, S)
        with pytest.raises(JWTException):
            core.parse_token(token, "refresh", S)

    def test_unknown_expected_type_never_matches(self):
        token, _ = core.issue_access_token("u", {}, S)
        with pytest.raises((JWTException, ValueError)):
            core.parse_token(token, "admin", S)

    def test_custom_claim_named_token_type_does_not_change_type(self):
        token, _ = core.issue_access_token("u", {"token_type": "refresh"}, S)
        with pytest.raises(JWTException):
            core.parse_token(token, "refresh", S)
        assert core.parse_token(token, "access", S).token_type == "access"

    @vuln
    @pytest.mark.parametrize("bad", ["admin", "", None, 1, "ACCESS"], ids=repr)
    def test_signed_payload_with_invalid_token_type_raises_library_exception(self, bad):
        with pytest.raises(JWTException):
            core.parse_token(craft(claims_dict(token_type=bad)), "access", S)

    # ---- input shape ---------------------------------------------------- #
    @pytest.mark.parametrize(
        "bad",
        [None, 123, 1.5, b"a.b.c", [], {}, object(), True],
        ids=lambda v: type(v).__name__,
    )
    def test_non_string_tokens_raise_decode_error(self, bad):
        with pytest.raises(TokenDecodeError):
            core.parse_token(bad, "access", S)

    @pytest.mark.parametrize("bad", ["", "a", "a.b", "a.b.c.d", "....", "a.b.c.d.e"])
    def test_wrong_segment_count_raises_decode_error(self, bad):
        with pytest.raises(TokenDecodeError):
            core.parse_token(bad, "access", S)

    @pytest.mark.parametrize("bad", ["..", " . . ", "...".replace(".", "A.")[:-1]])
    def test_empty_or_blank_segments_are_rejected(self, bad):
        with pytest.raises(JWTException):
            core.parse_token(bad, "access", S)

    @vuln
    @pytest.mark.parametrize(
        "bad",
        [
            "a.b.c",
            "é.é.é",
            "!!!.!!!.!!!",
            "\x00.\x00.\x00",
            "a.b.AAAA",
            "AAAA.AAAA.a",
            "====.====.====",
            "\n.\n.\n",
        ],
        ids=repr,
    )
    def test_garbage_tokens_raise_only_library_exceptions(self, bad):
        with pytest.raises(JWTException):
            core.parse_token(bad, "access", S)

    def test_bearer_prefix_is_not_accepted(self):
        with pytest.raises(JWTException):
            core.parse_token("Bearer " + craft(), "access", S)

    def test_leading_whitespace_is_not_accepted(self):
        with pytest.raises(JWTException):
            core.parse_token(" " + craft(), "access", S)

    @vuln
    @pytest.mark.parametrize(
        "suffix", ["\n", " ", "\r\n", "\t", "\x00", "=", "==", "!"], ids=repr
    )
    def test_trailing_garbage_after_signature_is_rejected(self, suffix):
        """Same token with different bytes must never validate (replay/denylist bypass)."""
        with pytest.raises(JWTException):
            core.parse_token(craft() + suffix, "access", S)

    @vuln
    def test_non_canonical_signature_encoding_is_rejected(self):
        h, p, s = craft().split(".")
        with pytest.raises(JWTException):
            core.parse_token(f"{h}.{p}.{flip_last_char_padding_bit(s)}", "access", S)

    @vuln
    @pytest.mark.parametrize("junk", ["\n", "!", " ", "+", "/"], ids=repr)
    def test_junk_characters_inside_signature_are_rejected(self, junk):
        h, p, s = craft().split(".")
        with pytest.raises(JWTException):
            core.parse_token(f"{h}.{p}.{s[:10]}{junk}{s[10:]}", "access", S)

    def test_huge_token_is_rejected_cleanly(self):
        with pytest.raises(JWTException):
            core.parse_token("A" * 2_000_000 + ".AAAA.AAAA", "access", S)

    # ---- signature ------------------------------------------------------ #
    def test_token_signed_with_other_secret_is_rejected(self):
        with pytest.raises(SignatureError):
            core.parse_token(craft(secret=OTHER_SECRET), "access", S)

    def test_token_from_other_settings_is_rejected(self):
        token, _ = core.issue_access_token("u", {}, make_settings(secret=OTHER_SECRET))
        with pytest.raises(SignatureError):
            core.parse_token(token, "access", S)

    @pytest.mark.parametrize(
        "changes",
        [
            {"sub": "admin"},
            {"claims": {"role": "admin"}},
            {"exp": 9_999_999_999},
            {"aud": "other"},
            {"token_type": "refresh"},
        ],
        ids=str,
    )
    def test_payload_tampering_is_detected(self, changes):
        token, _ = core.issue_access_token("alice", {"role": "user"}, S)
        with pytest.raises(SignatureError):
            core.parse_token(_tamper_payload(token, **changes), "access", S)

    def test_header_tampering_is_detected(self):
        h, p, s = craft().split(".")
        evil = b64(jdump({"alg": "HS256", "typ": "JWT", "x": 1}))
        with pytest.raises(SignatureError):
            core.parse_token(f"{evil}.{p}.{s}", "access", S)

    @pytest.mark.parametrize("position", [0, 5, 20, -1])
    def test_signature_bit_flips_are_detected(self, position):
        h, p, s = craft().split(".")
        idx = position % len(s)
        flipped = s[:idx] + ("A" if s[idx] != "A" else "B") + s[idx + 1 :]
        if position == -1:  # last char: change a *meaningful* bit, not padding bits
            flipped = s[:-1] + ("A" if s[-1] not in "AQgw" else "B")
        with pytest.raises(SignatureError):
            core.parse_token(f"{h}.{p}.{flipped}", "access", S)

    def test_signature_stripping_is_rejected(self):
        h, p, _ = craft().split(".")
        with pytest.raises(SignatureError):
            core.parse_token(f"{h}.{p}.", "access", S)

    def test_signature_swapped_between_tokens_is_rejected(self):
        a = craft(claims_dict(sub="alice"))
        b = craft(claims_dict(sub="bob"))
        ha, pa, _ = a.split(".")
        _, _, sb = b.split(".")
        with pytest.raises(SignatureError):
            core.parse_token(f"{ha}.{pa}.{sb}", "access", S)

    def test_payload_swapped_from_privileged_token_is_rejected(self):
        low = craft(claims_dict(sub="low"))
        high = craft(claims_dict(sub="root", claims={"role": "admin"}))
        h, _, s = low.split(".")
        _, p, _ = high.split(".")
        with pytest.raises(SignatureError):
            core.parse_token(f"{h}.{p}.{s}", "access", S)

    def test_signature_is_verified_before_payload_is_parsed(self):
        """No parsing oracle for unauthenticated input."""
        h = b64(jdump(HEADER))
        p = b64(b"not json at all")
        with pytest.raises(SignatureError):
            core.parse_token(f"{h}.{p}.{hmac_b64(h, p, OTHER_SECRET)}", "access", S)

    def test_signature_comparison_is_constant_time(self):
        with mock.patch.object(
            utils.hmac, "compare_digest", wraps=hmac.compare_digest
        ) as spy:
            core.parse_token(craft(), "access", S)
        assert spy.called

    def test_exceptions_do_not_leak_token_or_secret(self):
        token = craft(secret=OTHER_SECRET)
        with pytest.raises(JWTException) as exc:
            core.parse_token(token, "access", S)
        text = f"{exc.value!s} {exc.value!r}"
        assert SECRET not in text and token not in text

    # ---- algorithm / header attacks ------------------------------------ #
    def test_alg_none_with_empty_signature_is_rejected(self):
        h = b64(jdump({"alg": "none", "typ": "JWT"}))
        p = b64(jdump(claims_dict()))
        with pytest.raises(JWTException):
            core.parse_token(f"{h}.{p}.", "access", S)

    @vuln
    @pytest.mark.parametrize(
        "header",
        [
            {"alg": "none", "typ": "JWT"},
            {"alg": "None", "typ": "JWT"},
            {"alg": "NONE", "typ": "JWT"},
            {"alg": "hs256", "typ": "JWT"},
            {"alg": "HS384", "typ": "JWT"},
            {"alg": "HS512", "typ": "JWT"},
            {"alg": "RS256", "typ": "JWT"},
            {"alg": "ES256", "typ": "JWT"},
            {"alg": None, "typ": "JWT"},
            {"typ": "JWT"},
            {},
            {"alg": "HS256", "typ": "evil"},
            {"alg": "HS256", "typ": "JWT", "crit": ["exp"]},
        ],
        ids=lambda h: json.dumps(h),
    )
    def test_header_must_declare_hs256_even_with_valid_hmac(self, header):
        """The signature is valid, so only an explicit header check can stop
        algorithm-confusion / unsupported-critical-header tokens."""
        with pytest.raises(JWTException):
            core.parse_token(craft(header=header), "access", S)

    @vuln
    @pytest.mark.parametrize(
        "raw",
        [b"not json", b"[]", b'"HS256"', b"\xff\xfe", b"", b"null", b"[" * 100_000],
        ids=["text", "list", "string", "non-utf8", "empty", "null", "deep"],
    )
    def test_malformed_signed_header_is_rejected(self, raw):
        with pytest.raises(JWTException):
            core.parse_token(craft(raw_header=raw), "access", S)

    # ---- temporal claims ------------------------------------------------ #
    def test_expired_token_raises_expired_error(self):
        now = int(time.time())
        with pytest.raises(ExpiredTokenError):
            core.parse_token(
                craft(claims_dict(iat=now - 100, exp=now - 1)), "access", S
            )

    def test_token_expires_exactly_at_exp(self):
        with frozen(NOW):
            with pytest.raises(ExpiredTokenError):
                core.parse_token(craft(claims_dict(iat=NOW - 10, exp=NOW)), "access", S)

    def test_token_is_valid_until_the_last_second(self):
        with frozen(NOW):
            assert core.parse_token(
                craft(claims_dict(iat=NOW - 10, exp=NOW + 1)), "access", S
            )

    def test_token_expiry_follows_the_clock(self):
        with frozen(NOW):
            token, _ = core.issue_access_token("u", {}, S)
        with frozen(NOW + 899):
            assert core.verify_token(token, "access", S) is True
        with frozen(NOW + 900):
            with pytest.raises(ExpiredTokenError):
                core.parse_token(token, "access", S)

    def test_expired_token_with_valid_signature_cannot_be_revived_by_header(self):
        now = int(time.time())
        token = craft(claims_dict(iat=now - 100, exp=now - 1))
        h, p, s = token.split(".")
        with pytest.raises(JWTException):
            core.parse_token(f"{h}.{p}.{s}", "access", S)

    @vuln
    def test_token_issued_in_the_future_is_rejected(self):
        now = int(time.time())
        with pytest.raises(JWTException):
            core.parse_token(
                craft(claims_dict(iat=now + 3600, exp=now + 7200)), "access", S
            )

    @vuln
    def test_not_before_claim_is_enforced(self):
        now = int(time.time())
        token = craft({**claims_dict(), "nbf": now + 3600})
        with pytest.raises(JWTException):
            core.parse_token(token, "access", S)

    def test_exp_before_iat_in_signed_token_is_rejected(self):
        now = int(time.time())
        with pytest.raises(JWTException):
            core.parse_token(craft(claims_dict(iat=now, exp=now - 5)), "access", S)

    # ---- issuer / audience ---------------------------------------------- #
    def test_wrong_issuer_is_rejected(self):
        with pytest.raises(JWTException):
            core.parse_token(craft(claims_dict(iss="evil")), "access", S)

    def test_wrong_audience_is_rejected(self):
        with pytest.raises(JWTException):
            core.parse_token(craft(claims_dict(aud="other-service")), "access", S)

    def test_audience_list_is_not_treated_as_a_match(self):
        with pytest.raises(JWTException):
            core.parse_token(craft(claims_dict(aud=[S.aud])), "access", S)

    def test_token_for_other_audience_settings_is_rejected(self):
        token, _ = core.issue_access_token("u", {}, make_settings(aud="svc-b"))
        with pytest.raises(JWTException):
            core.parse_token(token, "access", S)

    def test_token_for_other_issuer_settings_is_rejected(self):
        token, _ = core.issue_access_token("u", {}, make_settings(iss="idp-2"))
        with pytest.raises(JWTException):
            core.parse_token(token, "access", S)

    # ---- hostile but correctly signed payloads --------------------------- #
    @vuln
    @pytest.mark.parametrize(
        "raw",
        [
            b"not json",
            b"",
            b"[]",
            b'"str"',
            b"123",
            b"null",
            b"\xff\xfe\x00",
            b"{}",
            b'{"jti":"x"}',
            b"[" * 100_000,
        ],
        ids=[
            "text",
            "empty",
            "list",
            "string",
            "number",
            "null",
            "non-utf8",
            "empty-object",
            "partial",
            "deep-nesting",
        ],
    )
    def test_malformed_signed_payloads_raise_library_exceptions(self, raw):
        with pytest.raises(JWTException):
            core.parse_token(craft(raw_payload=raw), "access", S)

    @vuln
    @pytest.mark.parametrize(
        "key", ["jti", "sub", "iat", "exp", "iss", "aud", "claims", "token_type"]
    )
    def test_signed_payload_missing_field_raises_library_exception(self, key):
        with pytest.raises(JWTException):
            core.parse_token(craft(claims_dict(**{key: _DROP})), "access", S)

    @vuln
    @pytest.mark.parametrize(
        "bad",
        [
            {"exp": "9999999999"},
            {"exp": True},
            {"exp": None},
            {"exp": [1]},
            {"iat": "1"},
            {"iat": None},
            {"sub": 123},
            {"sub": None},
            {"iss": None},
            {"aud": None},
            {"claims": [1]},
            {"claims": "x"},
            {"jti": None},
            {"jti": 123},
        ],
        ids=str,
    )
    def test_signed_payload_with_wrong_types_is_rejected(self, bad):
        with pytest.raises(JWTException):
            core.parse_token(craft(claims_dict(**bad)), "access", S)

    def test_unknown_top_level_fields_do_not_become_privileges(self):
        token = craft({**claims_dict(), "admin": True, "scope": "*"})
        p = core.parse_token(token, "access", S)
        assert not hasattr(p, "admin") and not hasattr(p, "scope")
        assert "admin" not in p.claims

    def test_parse_returns_independent_payload_per_call(self):
        token = craft()
        assert core.parse_token(token, "access", S) is not core.parse_token(
            token, "access", S
        )


# =========================================================================== #
# 6. core.verify_token
# =========================================================================== #
class TestVerifyToken:
    def test_true_for_valid_tokens(self):
        a, _ = core.issue_access_token("u", {}, S)
        r, _ = core.issue_refresh_token("u", {}, S)
        assert core.verify_token(a, "access", S) is True
        assert core.verify_token(r, "refresh", S) is True

    def test_default_type_is_access(self):
        a, _ = core.issue_access_token("u", {}, S)
        assert core.verify_token(a, settings=S) is True

    def test_false_for_wrong_type(self):
        a, _ = core.issue_access_token("u", {}, S)
        assert core.verify_token(a, "refresh", S) is False

    def test_false_for_expired(self):
        now = int(time.time())
        assert (
            core.verify_token(craft(claims_dict(iat=now - 9, exp=now - 1)), "access", S)
            is False
        )

    def test_false_for_wrong_secret_or_tampering(self):
        assert core.verify_token(craft(secret=OTHER_SECRET), "access", S) is False
        a, _ = core.issue_access_token("u", {}, S)
        assert core.verify_token(_tamper_payload(a, sub="root"), "access", S) is False

    @pytest.mark.parametrize(
        "bad", [None, 0, b"x.y.z", [], object()], ids=lambda v: type(v).__name__
    )
    def test_false_for_non_string_input(self, bad):
        assert core.verify_token(bad, "access", S) is False

    @vuln
    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "a.b.c",
            "é.é.é",
            "!!!.!!!.!!!",
            "....",
            "A" * 100 + ".B.C",
            "a.b.c\n",
            craft(raw_payload=b"{}"),
            craft(raw_payload=b"[]"),
            craft(raw_payload=b"\xff"),
            craft(claims_dict(exp="nan")),
            craft(claims_dict(token_type="x")),
        ],
        ids=lambda v: repr(v)[:25],
    )
    def test_never_raises_for_hostile_input(self, bad):
        """verify_token is a boolean gate: it must never leak an exception."""
        assert core.verify_token(bad, "access", S) is False


# =========================================================================== #
# 7. core.reissue_access_token
# =========================================================================== #
class TestReissue:
    @vuln
    def test_valid_refresh_token_yields_new_access_token(self):
        refresh, _ = core.issue_refresh_token("alice", {"role": "admin"}, S)
        token, jti = core.reissue_access_token(refresh, S)
        p = core.parse_token(token, "access", S)
        assert (
            p.token_type == "access"
            and p.sub == "alice"
            and p.claims == {"role": "admin"}
        )
        assert p.jti == jti

    @vuln
    def test_crafted_refresh_token_is_accepted(self):
        token, _ = core.reissue_access_token(
            craft(claims_dict(token_type="refresh")), S
        )
        assert core.verify_token(token, "access", S)

    @vuln
    def test_reissued_token_has_fresh_jti_and_lifetime(self):
        with frozen(NOW):
            refresh, rjti = core.issue_refresh_token("u", {}, S)
        with frozen(NOW + 100):
            token, jti = core.reissue_access_token(refresh, S)
            p = core.parse_token(token, "access", S)
        assert jti != rjti and p.iat == NOW + 100 and p.exp == NOW + 100 + 900

    @vuln
    def test_custom_settings_are_honoured(self):
        s2 = make_settings(secret=OTHER_SECRET, iss="idp-2", aud="svc-2", access_ttl=60)
        refresh, _ = core.issue_refresh_token("u", {}, s2)
        token, _ = core.reissue_access_token(refresh, s2)
        p = core.parse_token(token, "access", s2)
        assert (p.iss, p.aud) == ("idp-2", "svc-2") and p.exp - p.iat == 60
        assert core.verify_token(token, "access", S) is False

    @vuln
    def test_access_token_cannot_be_used_to_obtain_new_tokens(self):
        """Privilege escalation: an access token must never act as a refresh token,
        otherwise sessions can be extended forever with a stolen access token."""
        access, _ = core.issue_access_token("u", {}, S)
        with pytest.raises(JWTException):
            core.reissue_access_token(access, S)

    @vuln
    def test_reissued_token_cannot_be_chained_as_refresh_token(self):
        refresh, _ = core.issue_refresh_token("u", {}, S)
        token, _ = core.reissue_access_token(refresh, S)
        with pytest.raises(JWTException):
            core.reissue_access_token(token, S)

    @vuln
    def test_reissued_access_token_never_outlives_its_refresh_token(self):
        with frozen(NOW):
            refresh, _ = core.issue_refresh_token("u", {}, S)
        with frozen(NOW + 86400 - 60):  # one minute of refresh life left
            token, _ = core.reissue_access_token(refresh, S)
            p = core.parse_token(token, "access", S)
        assert p.exp <= NOW + 86400

    def test_expired_refresh_token_is_rejected(self):
        with frozen(NOW):
            refresh, _ = core.issue_refresh_token("u", {}, S)
        with frozen(NOW + 86400):
            with pytest.raises(ExpiredTokenError):
                core.reissue_access_token(refresh, S)

    def test_refresh_token_with_wrong_signature_is_rejected(self):
        with pytest.raises(SignatureError):
            core.reissue_access_token(
                craft(claims_dict(token_type="refresh"), secret=OTHER_SECRET), S
            )

    def test_refresh_token_for_wrong_issuer_or_audience_is_rejected(self):
        for bad in ({"iss": "evil"}, {"aud": "evil"}):
            with pytest.raises(JWTException):
                core.reissue_access_token(
                    craft(claims_dict(token_type="refresh", **bad)), S
                )

    @pytest.mark.parametrize("bad", [None, 123, b"x", "", "a.b", "a.b.c.d"], ids=repr)
    def test_non_token_input_is_rejected(self, bad):
        with pytest.raises(JWTException):
            core.reissue_access_token(bad, S)

    @vuln
    def test_tampered_refresh_token_is_rejected_by_signature(self):
        refresh, _ = core.issue_refresh_token("alice", {}, S)
        with pytest.raises(SignatureError):
            core.reissue_access_token(_tamper_payload(refresh, sub="root"), S)


# =========================================================================== #
# 8. Property-based / fuzz tests
# =========================================================================== #
class TestProperties:
    @given(sub=sub_st, claims=claims_st)
    def test_access_token_roundtrip(self, sub, claims):
        token, jti = core.issue_access_token(sub, claims, S)
        p = core.parse_token(token, "access", S)
        assert (p.sub, p.claims, p.jti) == (sub, claims, jti)

    @given(sub=sub_st, claims=claims_st)
    def test_refresh_token_roundtrip(self, sub, claims):
        token, _ = core.issue_refresh_token(sub, claims, S)
        p = core.parse_token(token, "refresh", S)
        assert (p.sub, p.claims, p.token_type) == (sub, claims, "refresh")

    @vuln
    @given(sub=sub_st, claims=claims_st)
    def test_reissue_roundtrip(self, sub, claims):
        refresh, _ = core.issue_refresh_token(sub, claims, S)
        token, _ = core.reissue_access_token(refresh, S)
        p = core.parse_token(token, "access", S)
        assert (p.sub, p.claims) == (sub, claims)

    @vuln
    @given(data=st.data())
    def test_any_single_character_mutation_invalidates_token(self, data):
        token, _ = core.issue_access_token("user", {"role": "user"}, S)
        idx = data.draw(st.integers(0, len(token) - 1))
        assume(token[idx] != ".")
        repl = data.draw(st.sampled_from(B64_ALPHABET))
        assume(repl != token[idx])
        mutated = token[:idx] + repl + token[idx + 1 :]
        assert core.verify_token(mutated, "access", S) is False

    @vuln
    @given(text=st.text(max_size=300))
    def test_verify_token_never_raises_on_arbitrary_text(self, text):
        assert core.verify_token(text, "access", S) is False

    @vuln
    @given(
        text=st.from_regex(
            r"[A-Za-z0-9_-]{0,80}\.[A-Za-z0-9_-]{0,80}\.[A-Za-z0-9_-]{0,80}",
            fullmatch=True,
        )
    )
    def test_verify_token_never_raises_on_token_shaped_text(self, text):
        assert core.verify_token(text, "access", S) is False

    @vuln
    @given(text=st.text(max_size=300))
    def test_parse_token_only_raises_library_exceptions(self, text):
        try:
            core.parse_token(text, "access", S)
        except JWTException:
            pass

    @given(raw=st.binary(max_size=200))
    def test_bytes_are_never_accepted_as_tokens(self, raw):
        assert core.verify_token(raw, "access", S) is False

    @given(other=secret_st)
    def test_token_is_never_valid_under_another_secret(self, other):
        assume(other != SECRET)
        token, _ = core.issue_access_token("u", {}, S)
        assert core.verify_token(token, "access", make_settings(secret=other)) is False

    @vuln
    @given(
        alg=st.sampled_from(
            [
                "none",
                "None",
                "NONE",
                "HS384",
                "HS512",
                "RS256",
                "ES256",
                "PS256",
                "EdDSA",
                "",
                "hs256",
            ]
        )
    )
    def test_no_algorithm_other_than_hs256_is_accepted(self, alg):
        assert (
            core.verify_token(craft(header={"alg": alg, "typ": "JWT"}), "access", S)
            is False
        )

    @given(sub=sub_st, other=sub_st)
    def test_subject_substitution_always_breaks_signature(self, sub, other):
        assume(sub != other)
        token, _ = core.issue_access_token(sub, {}, S)
        assert (
            core.verify_token(_tamper_payload(token, sub=other), "access", S) is False
        )

    @given(raw=st.binary(max_size=300))
    def test_b64_roundtrip(self, raw):
        assert utils.b64_decode(utils.b64_encode(raw)) == raw

    @given(raw=st.binary(max_size=300))
    def test_b64_output_is_unpadded_url_safe(self, raw):
        assert re.fullmatch(r"[A-Za-z0-9_-]*", utils.b64_encode(raw))

    @given(
        data=st.dictionaries(
            st.text(max_size=10),
            st.one_of(st.text(max_size=20), st.integers()),
            max_size=6,
        )
    )
    def test_json_roundtrip(self, data):
        assert utils.json_decode(utils.json_encode(data)) == data

    @given(
        h=st.text(max_size=40),
        p=st.text(max_size=40),
        secret=st.text(min_size=1, max_size=70),
    )
    def test_sign_matches_reference_hmac(self, h, p, secret):
        ref = hmac.new(secret.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
        assert utils.sign(h, p, secret) == ref

    @given(count=st.integers(2, 200))
    def test_generate_jti_is_collision_free(self, count):
        values = [utils.generate_jti() for _ in range(count)]
        assert len(set(values)) == count

    @given(sub=sub_st, claims=claims_st)
    def test_token_has_three_url_safe_segments(self, sub, claims):
        token, _ = core.issue_access_token(sub, claims, S)
        assert re.fullmatch(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", token)

    @given(
        sub=sub_st,
        iat=st.integers(0, 2**31),
        ttl=st.integers(0, 10**6),
        kind=st.sampled_from(["access", "refresh"]),
    )
    def test_payload_dict_roundtrip(self, sub, iat, ttl, kind):
        p = Payload(
            jti="j",
            sub=sub,
            iat=iat,
            exp=iat + ttl,
            iss="i",
            aud="a",
            claims={"k": sub},
            token_type=kind,
        )
        assert Payload.from_dict(p.to_dict()) == p

    @given(ttl=st.integers(min_value=1, max_value=10**6))
    def test_settings_accept_any_positive_ordered_ttls(self, ttl):
        assert make_settings(access_ttl=ttl, refresh_ttl=ttl + 1).access_ttl == ttl


# =========================================================================== #
# 9. Public API surface
# =========================================================================== #
class TestPublicApi:
    def test_all_exports_exactly_the_public_functions(self):
        assert set(jwt_pkg.__all__) == {
            "issue_access_token",
            "issue_refresh_token",
            "parse_token",
            "reissue_access_token",
            "verify_token",
        }

    def test_exported_names_are_callable(self):
        assert all(callable(getattr(jwt_pkg, name)) for name in jwt_pkg.__all__)

    @pytest.mark.parametrize(
        "internal", ["sign", "decode_token", "verify_signature", "b64_decode"]
    )
    def test_crypto_helpers_are_not_part_of_the_public_api(self, internal):
        assert internal not in jwt_pkg.__all__
        assert not hasattr(jwt_pkg, internal)

    def test_package_functions_are_the_core_functions(self):
        assert jwt_pkg.verify_token is core.verify_token
        assert jwt_pkg.issue_access_token is core.issue_access_token
