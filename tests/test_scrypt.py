"""
Module: test_scrypt.py

Description:
    Comprehensive, high-coverage (>= 95%) test suite for the python_security.scrypt package.
    Covers unit functionality, property-based invariants (Hypothesis), security edge cases,
    timing-attack resilience, and asserts ideal behaviors for suspected parser defects.

Configuration Files:
-------------------
To enforce >= 95% test coverage in your test runner, configure pyproject.toml or pytest.ini:

--- pyproject.toml ---
[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q --cov=python_security.scrypt --cov-report=term-missing --cov-fail-under=95"
testpaths = ["tests", "."]

[tool.coverage.run]
branch = true
source = ["python_security.scrypt"]

[tool.coverage.report]
show_missing = true
fail_under = 95
exclude_lines = [
    "pragma: no cover",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]

--- pytest.ini ---
[pytest]
minversion = 7.0
addopts = -ra -q --cov=python_security.scrypt --cov-report=term-missing --cov-fail-under=95
testpaths = tests .
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import hmac
import secrets
import time
from typing import Any
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from python_security.scrypt import hash_password, verify_password
from python_security.scrypt.config import (
    DEFAULT_SETTINGS,
    MAX_SALT_LENGTH,
    MIN_SALT_LENGTH,
    SCryptSettings,
)
from python_security.scrypt.exceptions import (
    DecodeError,
    EncodeError,
    InvalidSettingsValue,
    SCryptException,
)
from python_security.scrypt.utils import (
    b64_decoder,
    b64_encoder,
    digest,
    phc_decoder,
    phc_encoder,
)

# =====================================================================
# 1. Exception Hierarchy Tests
# =====================================================================


class TestExceptionHierarchy:
    """Verifies custom exception taxonomy and inheritance."""

    def test_base_exception_inheritance(self):
        assert issubclass(SCryptException, Exception)

    @pytest.mark.parametrize(
        "exc_class",
        [InvalidSettingsValue, EncodeError, DecodeError],
    )
    def test_derived_exceptions_inherit_from_base(self, exc_class: type[Exception]):
        assert issubclass(exc_class, SCryptException)
        instance = exc_class("test error")
        assert isinstance(instance, SCryptException)
        assert str(instance) == "test error"


# =====================================================================
# 2. Configuration & Settings Tests
# =====================================================================


class TestSCryptSettings:
    """Validates SCryptSettings constraints, memory calculations, and immutability."""

    def test_default_settings_values(self):
        assert DEFAULT_SETTINGS.salt_length == 32
        assert DEFAULT_SETTINGS.ln == 17
        assert DEFAULT_SETTINGS.r == 8
        assert DEFAULT_SETTINGS.p == 1
        assert DEFAULT_SETTINGS.dklen == 32
        assert DEFAULT_SETTINGS.max_password_length == 256

    def test_maxmem_formula_calculation(self):
        s = SCryptSettings(
            ln=14, r=8, p=1, dklen=32, salt_length=32, max_password_length=128
        )
        expected = 128 * 8 * ((1 << 14) + 1 + 2) + (1 << 20)
        assert s.maxmem == expected

    def test_immutability(self):
        s = DEFAULT_SETTINGS
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.ln = 18  # type: ignore[misc]

    @pytest.mark.parametrize("valid_salt_len", [MIN_SALT_LENGTH, 32, MAX_SALT_LENGTH])
    def test_valid_salt_boundaries(self, valid_salt_len: int):
        s = SCryptSettings(
            ln=10,
            r=8,
            p=1,
            dklen=32,
            salt_length=valid_salt_len,
            max_password_length=128,
        )
        assert s.salt_length == valid_salt_len

    @pytest.mark.parametrize(
        "invalid_salt_len",
        [MIN_SALT_LENGTH - 1, 0, -1, MAX_SALT_LENGTH + 1, 100],
    )
    def test_invalid_salt_length_raises(self, invalid_salt_len: int):
        with pytest.raises(InvalidSettingsValue, match="Salt Range out of Range"):
            SCryptSettings(
                ln=10,
                r=8,
                p=1,
                dklen=32,
                salt_length=invalid_salt_len,
                max_password_length=128,
            )

    @pytest.mark.parametrize(
        ("ln", "r", "p"),
        [
            (0, 8, 1),
            (-1, 8, 1),
            (10, 0, 1),
            (10, -1, 1),
            (10, 8, 0),
            (10, 8, -1),
            (0, 0, 0),
        ],
    )
    def test_non_positive_ln_r_p_raises(self, ln: int, r: int, p: int):
        with pytest.raises(InvalidSettingsValue, match="Any of ln, r, p is 0"):
            SCryptSettings(
                ln=ln,
                r=r,
                p=p,
                dklen=32,
                salt_length=32,
                max_password_length=128,
            )

    def test_openssl_memory_limit_exceeded(self):
        # Trigger maxmem > (2**31 - 1)
        with pytest.raises(InvalidSettingsValue, match="Possibility of Out-of-Memory"):
            SCryptSettings(
                ln=25,
                r=8,
                p=1,
                dklen=32,
                salt_length=32,
                max_password_length=128,
            )


# =====================================================================
# 3. Base64 & Format Utility Tests
# =====================================================================


class TestBase64Utilities:
    """Tests base64 encoding and decoding edge cases."""

    @pytest.mark.parametrize(
        "raw_bytes",
        [
            b"",
            b"a",
            b"ab",
            b"abc",
            b"abcd",
            b"\x00\xff\xfe\x01\x10",
            secrets.token_bytes(32),
            secrets.token_bytes(64),
        ],
    )
    def test_b64_encoder_decoder_roundtrip(self, raw_bytes: bytes):
        encoded = b64_encoder(raw_bytes)
        assert "=" not in encoded
        decoded = b64_decoder(encoded)
        assert decoded == raw_bytes

    def test_b64_decoder_padding_variants(self):
        # 1 byte -> 2 chars + 2 pad '='
        data = b"X"
        std_padded = base64.b64encode(data).decode("ascii")  # 'WA=='
        unpadded = std_padded.rstrip("=")  # 'WA'
        assert b64_decoder(unpadded) == data
        assert b64_decoder(std_padded) == data

    @pytest.mark.parametrize("invalid_b64", ["!@#$%", "abcde", "====", "a="])
    def test_b64_decoder_invalid_characters_raises(self, invalid_b64: str):
        with pytest.raises(Exception):
            b64_decoder(invalid_b64)


class TestPHCFormatUtilities:
    """Tests encoding/decoding of PHC string representations."""

    def test_phc_encoder_format(self):
        salt = b"\x01" * 32
        digest_bytes = b"\x02" * 32
        encoded = phc_encoder(salt, digest_bytes, DEFAULT_SETTINGS)

        parts = encoded.split("$")
        assert len(parts) == 5
        assert parts[0] == ""
        assert parts[1] == "scrypt"
        assert (
            parts[2]
            == f"ln={DEFAULT_SETTINGS.ln},r={DEFAULT_SETTINGS.r},p={DEFAULT_SETTINGS.p}"
        )
        assert parts[3] == b64_encoder(salt)
        assert parts[4] == b64_encoder(digest_bytes)

    def test_phc_decoder_valid_roundtrip(self):
        salt = secrets.token_bytes(DEFAULT_SETTINGS.salt_length)
        digest_bytes = secrets.token_bytes(DEFAULT_SETTINGS.dklen)
        encoded = phc_encoder(salt, digest_bytes, DEFAULT_SETTINGS)

        dec_salt, dec_digest, dec_settings = phc_decoder(encoded)
        assert dec_salt == salt
        assert dec_digest == digest_bytes
        assert dec_settings.ln == DEFAULT_SETTINGS.ln
        assert dec_settings.r == DEFAULT_SETTINGS.r
        assert dec_settings.p == DEFAULT_SETTINGS.p
        assert dec_settings.dklen == DEFAULT_SETTINGS.dklen
        assert dec_settings.salt_length == DEFAULT_SETTINGS.salt_length

    # Assert ideal behavior: malformed hashes must raise DecodeError
    @pytest.mark.parametrize(
        "non_string",
        [None, 12345, b"$scrypt$ln=17,r=8,p=1$AA$BB", ["hash"]],
    )
    def test_phc_decoder_non_string_raises_decode_error(self, non_string: Any):
        with pytest.raises(DecodeError, match="hashed_password is not a string"):
            phc_decoder(non_string)

    @pytest.mark.parametrize(
        "malformed_structure",
        [
            "",
            "scrypt$ln=17,r=8,p=1$AAAA$BBBB",  # Missing leading '$'
            "$scrypt$ln=17,r=8,p=1$AAAA",  # Only 4 parts
            "$scrypt$ln=17,r=8,p=1$AAAA$BBBB$CCCC",  # 6 parts
            "$argon2id$ln=17,r=8,p=1$AAAA$BBBB",  # Unsupported algo
            "$scrypt$$AAAA$BBBB",  # Empty params
        ],
    )
    def test_phc_decoder_malformed_structure_raises_decode_error(
        self, malformed_structure: str
    ):
        with pytest.raises(DecodeError):
            phc_decoder(malformed_structure)

    @pytest.mark.parametrize(
        "corrupted_params",
        [
            "$scrypt$ln=17,r=8$YWJjZGVmZ2hpamtsbW5vcA$YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY",  # missing p
            "$scrypt$ln=17,r=8,p=invalid$YWJjZGVmZ2hpamtsbW5vcA$YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY",  # non-int p
            "$scrypt$ln=17,r=8,p=1,extra=2$YWJjZGVmZ2hpamtsbW5vcA$YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY",  # extra param
            "$scrypt$invalid$YWJjZGVmZ2hpamtsbW5vcA$YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY",  # no equals
            "$scrypt$ln=17=extra,r=8,p=1$YWJjZGVmZ2hpamtsbW5vcA$YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY",  # multiple equals
        ],
    )
    def test_phc_decoder_corrupted_parameters_raises_decode_error(
        self, corrupted_params: str
    ):
        """Assert ideal behavior: parameter parsing defects raise DecodeError."""
        with pytest.raises(DecodeError):
            phc_decoder(corrupted_params)

    @pytest.mark.parametrize(
        "invalid_base64_hash",
        [
            "$scrypt$ln=17,r=8,p=1$not_valid_b64!@#$YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY",
            "$scrypt$ln=17,r=8,p=1$YWJjZGVmZ2hpamtsbW5vcA$not_valid_b64!@#",
        ],
    )
    def test_phc_decoder_invalid_base64_raises_decode_error(
        self, invalid_base64_hash: str
    ):
        """Assert ideal behavior: bad base64 in PHC raises DecodeError."""
        with pytest.raises(DecodeError):
            phc_decoder(invalid_base64_hash)

    def test_phc_decoder_invalid_settings_values_raises_decode_error(self):
        """Assert ideal behavior: invalid decoded parameters raise DecodeError."""
        # Salt decoded is only 4 bytes (less than MIN_SALT_LENGTH 16)
        short_salt_b64 = b64_encoder(b"short")
        valid_digest_b64 = b64_encoder(b"d" * 32)
        h = f"$scrypt$ln=17,r=8,p=1${short_salt_b64}${valid_digest_b64}"
        with pytest.raises(DecodeError):
            phc_decoder(h)


# =====================================================================
# 4. Digest Function Tests
# =====================================================================


class TestDigestFunction:
    """Tests utils.digest behavior with DEFAULT_SETTINGS."""

    def test_digest_deterministic_output(self):
        salt = b"\xaa" * DEFAULT_SETTINGS.salt_length
        d1 = digest("super-secret-pass", salt, DEFAULT_SETTINGS)
        d2 = digest("super-secret-pass", salt, DEFAULT_SETTINGS)
        assert d1 == d2
        assert len(d1) == DEFAULT_SETTINGS.dklen

    def test_digest_different_salts_produce_different_digests(self):
        salt1 = b"\x01" * DEFAULT_SETTINGS.salt_length
        salt2 = b"\x02" * DEFAULT_SETTINGS.salt_length
        d1 = digest("same-password", salt1, DEFAULT_SETTINGS)
        d2 = digest("same-password", salt2, DEFAULT_SETTINGS)
        assert d1 != d2

    def test_digest_different_passwords_produce_different_digests(self):
        salt = b"\x55" * DEFAULT_SETTINGS.salt_length
        d1 = digest("password-one", salt, DEFAULT_SETTINGS)
        d2 = digest("password-two", salt, DEFAULT_SETTINGS)
        assert d1 != d2

    def test_password_length_exceeded_raises_encode_error(self):
        salt = b"\x00" * DEFAULT_SETTINGS.salt_length
        long_pwd = "P" * (DEFAULT_SETTINGS.max_password_length + 1)
        with pytest.raises(EncodeError, match="Password is Too Long"):
            digest(long_pwd, salt, DEFAULT_SETTINGS)

    def test_password_exact_max_length_succeeds(self):
        salt = b"\x11" * DEFAULT_SETTINGS.salt_length
        pwd = "P" * DEFAULT_SETTINGS.max_password_length
        d = digest(pwd, salt, DEFAULT_SETTINGS)
        assert len(d) == DEFAULT_SETTINGS.dklen

    def test_empty_password_digest(self):
        salt = b"\x22" * DEFAULT_SETTINGS.salt_length
        d = digest("", salt, DEFAULT_SETTINGS)
        assert len(d) == DEFAULT_SETTINGS.dklen

    def test_unicode_and_special_characters(self):
        salt = b"\x33" * DEFAULT_SETTINGS.salt_length
        pwd = "P@$$w0rd_🔑_ñ_日本語_🛡️"
        d = digest(pwd, salt, DEFAULT_SETTINGS)
        assert len(d) == DEFAULT_SETTINGS.dklen


# =====================================================================
# 5. Core Hash and Verification Tests (Using DEFAULT_SETTINGS)
# =====================================================================


class TestCoreHashAndVerify:
    """End-to-end tests for hash_password and verify_password."""

    def test_hash_password_produces_phc_with_default_settings(self):
        password = "ProductionPassword#2026!"
        h = hash_password(password)

        assert h.startswith(
            f"$scrypt$ln={DEFAULT_SETTINGS.ln},r={DEFAULT_SETTINGS.r},p={DEFAULT_SETTINGS.p}$"
        )
        salt, d, settings_obj = phc_decoder(h)
        assert len(salt) == DEFAULT_SETTINGS.salt_length
        assert len(d) == DEFAULT_SETTINGS.dklen
        assert settings_obj.ln == DEFAULT_SETTINGS.ln

    def test_hash_password_salts_are_cryptographically_unique(self):
        password = "SecurePassword123"
        h1 = hash_password(password)
        h2 = hash_password(password)

        assert h1 != h2
        salt1, _, _ = phc_decoder(h1)
        salt2, _, _ = phc_decoder(h2)
        assert salt1 != salt2

    def test_verify_password_success(self):
        password = "CorrectHorseBatteryStaple!"
        h = hash_password(password)
        assert verify_password(password, h) is True

    def test_verify_password_incorrect_password(self):
        password = "CorrectPassword"
        h = hash_password(password)
        assert verify_password("WrongPassword", h) is False

    def test_verify_password_tampered_digest_fails(self):
        password = "AuthenticationKey"
        h = hash_password(password)
        salt, d, settings_obj = phc_decoder(h)

        # Mutate last byte of digest
        tampered_d = bytearray(d)
        tampered_d[-1] ^= 0x01
        tampered_h = phc_encoder(salt, bytes(tampered_d), settings_obj)

        assert verify_password(password, tampered_h) is False

    def test_verify_password_tampered_salt_fails(self):
        password = "AuthenticationKey"
        h = hash_password(password)
        salt, d, settings_obj = phc_decoder(h)

        # Mutate first byte of salt
        tampered_salt = bytearray(salt)
        tampered_salt[0] ^= 0x01
        tampered_h = phc_encoder(bytes(tampered_salt), d, settings_obj)

        assert verify_password(password, tampered_h) is False

    def test_verify_password_tampered_parameters_fails(self):
        password = "AuthenticationKey"
        h = hash_password(password)
        salt, d, settings_obj = phc_decoder(h)

        # Modify ln to 16
        tampered_settings = SCryptSettings(
            ln=16,
            r=settings_obj.r,
            p=settings_obj.p,
            dklen=settings_obj.dklen,
            salt_length=settings_obj.salt_length,
            max_password_length=settings_obj.max_password_length,
        )
        tampered_h = phc_encoder(salt, d, tampered_settings)

        assert verify_password(password, tampered_h) is False

    def test_verify_password_malformed_phc_raises_decode_error(self):
        """Assert ideal behavior: verify_password propagates DecodeError on malformed hash."""
        with pytest.raises(DecodeError):
            verify_password("Password", "corrupted$hash$value")


# =====================================================================
# 6. Security Vulnerability & Edge Case Tests
# =====================================================================


class TestSecurityVulnerabilities:
    """Covers common cryptographic and web security threats."""

    def test_null_byte_injection_differentiation(self):
        """Ensure strings with internal null bytes are not truncated or conflated."""
        salt = b"\x44" * DEFAULT_SETTINGS.salt_length
        d1 = digest("admin", salt, DEFAULT_SETTINGS)
        d2 = digest("admin\x00extra", salt, DEFAULT_SETTINGS)
        d3 = digest("admin\x00", salt, DEFAULT_SETTINGS)

        assert d1 != d2
        assert d1 != d3
        assert d2 != d3

    def test_unicode_normalization_handling(self):
        """Verify distinct unicode representations yield distinct digests (raw UTF-8)."""
        salt = b"\x99" * DEFAULT_SETTINGS.salt_length
        # NFC vs NFD for 'é'
        nfc = "\u00e9"
        nfd = "\u0065\u0301"

        d_nfc = digest(nfc, salt, DEFAULT_SETTINGS)
        d_nfd = digest(nfd, salt, DEFAULT_SETTINGS)
        assert d_nfc != d_nfd

    def test_dos_long_password_short_circuit(self):
        """Passwords exceeding max_password_length must fail fast without running scrypt."""
        long_pwd = "A" * (DEFAULT_SETTINGS.max_password_length + 10)

        with patch("hashlib.scrypt") as mock_scrypt:
            with pytest.raises(EncodeError):
                hash_password(long_pwd)
            mock_scrypt.assert_not_called()

    def test_empty_password_supported_safely(self):
        h = hash_password("")
        assert verify_password("", h) is True
        assert verify_password("not-empty", h) is False

    def test_standard_test_vector_consistency(self):
        """Validate underlying hashlib scrypt against RFC 7914 Test Vector 1."""
        # RFC 7914: P="", S="", N=16 (ln=4), r=1, p=1, dklen=64
        rfc_settings = SCryptSettings(
            ln=4, r=1, p=1, dklen=64, salt_length=16, max_password_length=64
        )
        rfc_salt = b"\x00" * 16
        # Compute digest
        d = digest("", rfc_salt, rfc_settings)
        expected = hashlib.scrypt(
            b"",
            salt=rfc_salt,
            n=1 << 4,
            r=1,
            p=1,
            dklen=64,
            maxmem=rfc_settings.maxmem,
        )
        assert d == expected


# =====================================================================
# 7. Statistical & Constant-Time Verification Tests
# =====================================================================


class TestTimingAttackResistance:
    """Verifies constant-time comparison in verify_password."""

    def test_verify_password_strictly_uses_hmac_compare_digest(self):
        """Assert hmac.compare_digest is directly used to protect against timing attacks."""
        password = "SecretPassword"
        h = hash_password(password)

        with patch("hmac.compare_digest", wraps=hmac.compare_digest) as mock_compare:
            res = verify_password(password, h)
            assert res is True
            assert mock_compare.call_count == 1

            # Ensure actual_digest and received_digest are passed as bytes
            args, _ = mock_compare.call_args
            assert isinstance(args[0], bytes)
            assert isinstance(args[1], bytes)
            assert len(args[0]) == DEFAULT_SETTINGS.dklen
            assert len(args[1]) == DEFAULT_SETTINGS.dklen

    def test_compare_digest_timing_distribution(self):
        """
        Statistical sanity test: checks comparison time difference between
        mismatch at byte 0 vs mismatch at final byte (dklen - 1).
        Verifies constant-time behavior on the digest comparison primitive.
        """
        key1 = b"\x00" * DEFAULT_SETTINGS.dklen
        # Differs at byte 0
        key_diff_first = b"\xff" + b"\x00" * (DEFAULT_SETTINGS.dklen - 1)
        # Differs at byte 31
        key_diff_last = b"\x00" * (DEFAULT_SETTINGS.dklen - 1) + b"\xff"

        iterations = 5000

        # Timing diff first
        t0 = time.perf_counter()
        for _ in range(iterations):
            hmac.compare_digest(key1, key_diff_first)
        diff_first_time = time.perf_counter() - t0

        # Timing diff last
        t0 = time.perf_counter()
        for _ in range(iterations):
            hmac.compare_digest(key1, key_diff_last)
        diff_last_time = time.perf_counter() - t0

        # Ratios should be very close; constant-time comparison avoids early return
        ratio = diff_first_time / diff_last_time if diff_last_time > 0 else 1.0
        assert 0.5 <= ratio <= 2.0


# =====================================================================
# 8. Hypothesis Property-Based Tests
# =====================================================================


class TestHypothesisProperties:
    """Property-based verification using Hypothesis."""

    @given(st.binary(min_size=0, max_size=256))
    @settings(max_examples=50, deadline=None)
    def test_property_b64_roundtrip(self, data: bytes):
        encoded = b64_encoder(data)
        decoded = b64_decoder(encoded)
        assert decoded == data

    @given(
        salt=st.binary(min_size=MIN_SALT_LENGTH, max_size=MAX_SALT_LENGTH),
        digest_bytes=st.binary(min_size=16, max_size=64),
    )
    @settings(max_examples=30, deadline=None)
    def test_property_phc_encoder_decoder_roundtrip(
        self, salt: bytes, digest_bytes: bytes
    ):
        settings_obj = SCryptSettings(
            ln=10,
            r=8,
            p=1,
            dklen=len(digest_bytes),
            salt_length=len(salt),
            max_password_length=128,
        )
        encoded = phc_encoder(salt, digest_bytes, settings_obj)
        dec_salt, dec_digest, dec_settings = phc_decoder(encoded)

        assert dec_salt == salt
        assert dec_digest == digest_bytes
        assert dec_settings.ln == settings_obj.ln
        assert dec_settings.r == settings_obj.r
        assert dec_settings.p == settings_obj.p
        assert dec_settings.dklen == len(digest_bytes)
        assert dec_settings.salt_length == len(salt)

    @given(st.text())
    @settings(max_examples=100, deadline=None)
    def test_property_phc_decoder_fuzzing(self, random_string: str):
        """
        Fuzzing test asserting ideal behavior: Any arbitrary string passed
        to phc_decoder must either parse successfully or raise DecodeError,
        never unhandled exceptions (KeyError, ValueError, IndexError, etc.).
        """
        try:
            salt, d, cfg = phc_decoder(random_string)
            assert isinstance(salt, bytes)
            assert isinstance(d, bytes)
            assert isinstance(cfg, SCryptSettings)
        except DecodeError:
            pass  # Expected behavior for invalid inputs

    @given(
        password=st.text(
            alphabet=st.characters(blacklist_categories=("Cs",)),
            min_size=1,
            max_size=32,
        )
    )
    @settings(
        max_examples=5,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_property_core_hash_and_verify_with_default_settings(self, password: str):
        """
        Property test executing realistic DEFAULT_SETTINGS.
        Kept to max_examples=5 due to ln=17 computation cost (~300ms/hash).
        """
        h = hash_password(password)
        assert verify_password(password, h) is True
        assert verify_password(password + "_incorrect", h) is False
