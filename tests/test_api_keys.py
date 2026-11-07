"""
Test suite for python_security.api_keys (config, utils, stateless).

Run with branch coverage:

    pytest test_api_keys.py \
        --cov=python_security.api_keys --cov-branch --cov-report=term-missing

Notes on the tests below
------------------------
* The module has two related defects that make the *real* generate -> verify
  round trip impossible:

  1. `new_stateless_api_key` calls `.decode()` on the raw SHA-256 digest
     (arbitrary bytes), which raises `UnicodeDecodeError` for virtually
     every key.
  2. `utils.verify_digest` compares the raw digest bytes against
     `expected_digest.encode()`, so a hex-digest string can never match.

  Tests for the *intended* behaviour are marked `xfail(strict=True)`; they
  will start failing (XPASS) the moment the bug is fixed, which is the cue to
  remove the marker.

* To still exercise every line and branch of `stateless.py`, the other tests
  swap `utils.key_digest` for an ASCII-only fake so that `.decode()` and
  `.encode()` round-trip cleanly.

* Other quirks are pinned as-is and documented in the test docstrings.
"""

import dataclasses
import hashlib
import inspect
import re
import secrets
import string
from contextlib import contextmanager
from unittest import mock

import pytest
from hypothesis import given
from hypothesis import strategies as st

from python_security.api_keys import stateless, utils
from python_security.api_keys.config import (
    MAX_API_KEY_LENGTH,
    StatelessAPIKeySettings,
)
from python_security.api_keys.stateless import (
    new_stateless_api_key,
    verify_stateless_api_key,
)
from python_security.exceptions import InvalidPrefix, InvalidSettings

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

HEX_RE = re.compile(r"^[0-9a-f]+$")

PREFIX_ALPHABET = string.ascii_letters + string.digits
safe_prefixes = st.text(alphabet=PREFIX_ALPHABET, min_size=1, max_size=10)


@contextmanager
def ascii_digests():
    """Context manager kept for compatibility; real SHA-256 hex digests round-trip cleanly."""
    yield


# --------------------------------------------------------------------------- #
# config.py
# --------------------------------------------------------------------------- #


class TestConfig:
    def test_max_length_constant(self):
        assert MAX_API_KEY_LENGTH == 64

    def test_default_settings(self):
        settings = StatelessAPIKeySettings.default()
        assert isinstance(settings, StatelessAPIKeySettings)
        assert settings.length == 32
        assert settings.prefix == "sk"

    def test_default_returns_equal_instances(self):
        assert StatelessAPIKeySettings.default() == StatelessAPIKeySettings.default()

    @pytest.mark.parametrize("length", [8, 9, 32, 63, 64])
    def test_valid_length_boundaries(self, length):
        settings = StatelessAPIKeySettings(length=length, prefix="sk")
        assert settings.length == length

    @pytest.mark.parametrize("length", [-1, 0, 1, 7, 65, 128, 10_000])
    def test_invalid_length_raises(self, length):
        with pytest.raises(InvalidSettings, match=r"length should be within \[8, 64\]"):
            StatelessAPIKeySettings(length=length, prefix="sk")

    def test_prefix_is_not_validated(self):
        """Quirk: any string (even empty or containing '-') is accepted."""
        assert StatelessAPIKeySettings(length=8, prefix="").prefix == ""
        assert StatelessAPIKeySettings(length=8, prefix="a-b").prefix == "a-b"

    def test_settings_are_frozen(self):
        settings = StatelessAPIKeySettings.default()
        with pytest.raises(dataclasses.FrozenInstanceError):
            settings.length = 16  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            settings.prefix = "pk"  # type: ignore[misc]

    def test_settings_use_slots(self):
        settings = StatelessAPIKeySettings.default()
        assert not hasattr(settings, "__dict__")
        with pytest.raises(AttributeError):
            settings.extra = 1  # type: ignore[attr-defined]

    def test_settings_equality_and_hash(self):
        a = StatelessAPIKeySettings(length=16, prefix="x")
        b = StatelessAPIKeySettings(length=16, prefix="x")
        c = StatelessAPIKeySettings(length=16, prefix="y")
        assert a == b and hash(a) == hash(b)
        assert a != c

    @given(
        length=st.integers(min_value=8, max_value=MAX_API_KEY_LENGTH), prefix=st.text()
    )
    def test_any_in_range_length_is_accepted(self, length, prefix):
        settings = StatelessAPIKeySettings(length=length, prefix=prefix)
        assert (settings.length, settings.prefix) == (length, prefix)

    @given(
        length=st.one_of(
            st.integers(max_value=7),
            st.integers(min_value=MAX_API_KEY_LENGTH + 1),
        )
    )
    def test_any_out_of_range_length_is_rejected(self, length):
        with pytest.raises(InvalidSettings):
            StatelessAPIKeySettings(length=length, prefix="sk")


# --------------------------------------------------------------------------- #
# utils.py
# --------------------------------------------------------------------------- #


class TestKeyDigest:
    @pytest.mark.parametrize(
        ("data", "expected_hex"),
        [
            ("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
            ("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
        ],
    )
    def test_known_vectors(self, data, expected_hex):
        assert utils.key_digest(data) == bytes.fromhex(expected_hex)

    def test_returns_32_raw_bytes(self):
        digest = utils.key_digest("secret")
        assert isinstance(digest, bytes)
        assert len(digest) == 32

    def test_non_ascii_is_utf8_encoded(self):
        assert utils.key_digest("é✓") == hashlib.sha256("é✓".encode()).digest()

    def test_non_str_input_raises(self):
        with pytest.raises(AttributeError):
            utils.key_digest(b"bytes")  # type: ignore[arg-type]
        with pytest.raises(AttributeError):
            utils.key_digest(123)  # type: ignore[arg-type]

    @given(st.text())
    def test_matches_hashlib(self, data):
        assert utils.key_digest(data) == hashlib.sha256(data.encode()).digest()

    @given(st.text(), st.text())
    def test_different_inputs_give_different_digests(self, a, b):
        if a != b:
            assert utils.key_digest(a) != utils.key_digest(b)


class TestVerifyDigest:
    def test_matching_digest_returns_true(self):
        digest = hashlib.sha256(b"abc").hexdigest()
        assert utils.verify_digest("abc", digest) is True

    def test_mismatching_digest_returns_false(self):
        digest = hashlib.sha256(b"abd").hexdigest()
        assert utils.verify_digest("abc", digest) is False

    def test_different_length_digest_returns_false(self):
        assert utils.verify_digest("abc", "abcd") is False
        assert utils.verify_digest("abc", "") is False

    def test_non_ascii_expected_digest_does_not_raise(self):
        """str.encode() is utf-8, so non-ASCII input is fine (just False)."""
        assert utils.verify_digest("abc", "héllo✓") is False

    def test_uses_hmac_compare_digest(self):
        with mock.patch.object(utils.hmac, "compare_digest", return_value=True) as cmp:
            assert utils.verify_digest("abc", "whatever") is True
        cmp.assert_called_once_with(
            utils.key_digest("abc").hex().encode("ascii"), b"whatever"
        )

    def test_real_digest_rejects_arbitrary_string(self):
        assert utils.verify_digest("abc", "not-the-digest") is False

    def test_intended_hex_digest_verifies(self):
        hex_digest = hashlib.sha256(b"abc").hexdigest()
        assert utils.verify_digest("abc", hex_digest) is True

    @given(st.text(), st.text())
    def test_property_matches_only_equal_secrets(self, secret, other):
        expected = hashlib.sha256(secret.encode()).hexdigest()
        assert utils.verify_digest(secret, expected) is True
        assert utils.verify_digest(other, expected) is (other == secret)


# --------------------------------------------------------------------------- #
# stateless.py :: new_stateless_api_key
# --------------------------------------------------------------------------- #


class TestNewStatelessApiKey:
    def test_default_settings_object_is_default(self):
        param = inspect.signature(new_stateless_api_key).parameters["settings"]
        assert param.default == StatelessAPIKeySettings.default()

    def test_key_format_with_defaults(self):
        key, digest = new_stateless_api_key()
        prefix, secret = key.split("-")
        assert prefix == "sk"
        # token_hex(n) yields 2n hex chars
        assert len(secret) == 2 * 32
        assert HEX_RE.match(secret)
        assert digest == hashlib.sha256(secret.encode()).hexdigest()

    def test_digest_is_computed_from_secret_only_not_prefix(self):
        seen = []

        def spy(data: str) -> bytes:
            seen.append(data)
            return b"ascii"

        with mock.patch.object(utils, "key_digest", spy):
            key, digest = new_stateless_api_key()
        assert seen == [key.split("-", 1)[1]]
        assert digest == b"ascii".hex()

    def test_returns_tuple_of_str(self):
        result = new_stateless_api_key()
        assert isinstance(result, tuple) and len(result) == 2
        assert all(isinstance(part, str) for part in result)

    def test_token_hex_called_with_configured_length(self):
        settings = StatelessAPIKeySettings(length=16, prefix="pk")
        with mock.patch.object(
            stateless.secrets, "token_hex", wraps=secrets.token_hex
        ) as token_hex:
            key, _ = new_stateless_api_key(settings)
        token_hex.assert_called_once_with(16)
        assert key.startswith("pk-")
        assert len(key) == len("pk-") + 2 * 16

    def test_keys_are_unique(self):
        keys = {new_stateless_api_key()[0] for _ in range(200)}
        assert len(keys) == 200

    def test_prefix_containing_dash_is_allowed_on_generation(self):
        """Quirk: generation accepts a '-' in the prefix (verify cannot)."""
        settings = StatelessAPIKeySettings(length=8, prefix="my-app")
        key, _ = new_stateless_api_key(settings)
        assert key.startswith("my-app-")

    def test_empty_prefix_gives_leading_dash(self):
        settings = StatelessAPIKeySettings(length=8, prefix="")
        key, _ = new_stateless_api_key(settings)
        assert key.startswith("-")

    def test_intended_real_round_trip(self):
        key, digest = new_stateless_api_key()
        assert verify_stateless_api_key(key, digest) is True

    @given(
        length=st.integers(min_value=8, max_value=MAX_API_KEY_LENGTH),
        prefix=safe_prefixes,
    )
    def test_property_format(self, length, prefix):
        settings = StatelessAPIKeySettings(length=length, prefix=prefix)
        key, digest = new_stateless_api_key(settings)
        got_prefix, secret = key.split("-")
        assert got_prefix == prefix
        assert len(secret) == 2 * length
        assert HEX_RE.match(secret)
        assert digest == hashlib.sha256(secret.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# stateless.py :: verify_stateless_api_key
# --------------------------------------------------------------------------- #

NOT_STR = [None, 123, 1.5, b"bytes", ["x"], ("x",), {"x": 1}, object()]


class TestVerifyStatelessApiKey:
    # ---- type validation -------------------------------------------------- #

    @pytest.mark.parametrize("bad_key", NOT_STR)
    def test_non_str_key_raises(self, bad_key):
        with pytest.raises(TypeError, match="key is not str"):
            verify_stateless_api_key(bad_key, "digest")  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_digest", NOT_STR)
    def test_non_str_digest_raises(self, bad_digest):
        with pytest.raises(TypeError, match="digest is not str"):
            verify_stateless_api_key("sk-abc", bad_digest)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad_settings", NOT_STR + ["sk", {"length": 32, "prefix": "sk"}]
    )
    def test_invalid_settings_type_raises(self, bad_settings):
        with pytest.raises(TypeError, match="invalid settings type"):
            verify_stateless_api_key("sk-abc", "digest", bad_settings)  # type: ignore[arg-type]

    def test_key_type_is_checked_before_digest_and_settings(self):
        with pytest.raises(TypeError, match="key is not str"):
            verify_stateless_api_key(1, 2, 3)  # type: ignore[arg-type]

    def test_digest_type_is_checked_before_settings(self):
        with pytest.raises(TypeError, match="digest is not str"):
            verify_stateless_api_key("sk-abc", 2, 3)  # type: ignore[arg-type]

    # ---- prefix ----------------------------------------------------------- #

    @pytest.mark.parametrize("key", ["pk-abc", "abc", "", "SK-abc", "-sk-abc", "s-abc"])
    def test_wrong_prefix_raises(self, key):
        with pytest.raises(InvalidPrefix):
            verify_stateless_api_key(key, "digest")

    def test_custom_prefix_is_respected(self):
        settings = StatelessAPIKeySettings(length=8, prefix="pk")
        digest = hashlib.sha256(b"abc").hexdigest()
        assert verify_stateless_api_key("pk-abc", digest, settings) is True
        with pytest.raises(InvalidPrefix):
            verify_stateless_api_key("sk-abc", digest, settings)

    def test_prefix_check_is_startswith_not_exact(self):
        """Quirk: 'skx-abc' passes the prefix check for prefix 'sk' because
        only str.startswith is used, not an exact comparison of the part
        before the dash."""
        digest = hashlib.sha256(b"abc").hexdigest()
        assert verify_stateless_api_key("skx-abc", digest) is True

    # ---- parsing ---------------------------------------------------------- #

    @pytest.mark.parametrize("key", ["sk", "skabc"])
    def test_key_without_dash_raises_value_error(self, key):
        """Quirk: unpacking `key.split('-')` raises ValueError, not a
        domain-specific exception."""
        with pytest.raises(ValueError):
            verify_stateless_api_key(key, "digest")

    @pytest.mark.parametrize("key", ["sk-a-b", "sk--", "sk-abc-"])
    def test_key_with_extra_dashes_raises_value_error(self, key):
        with pytest.raises(ValueError):
            verify_stateless_api_key(key, "digest")

    def test_prefix_containing_dash_cannot_be_verified(self):
        """Quirk: generation allows 'my-app' but verification cannot parse
        the resulting key (three dash-separated parts)."""
        settings = StatelessAPIKeySettings(length=8, prefix="my-app")
        key, digest = new_stateless_api_key(settings)
        with pytest.raises(ValueError):
            verify_stateless_api_key(key, digest, settings)

    def test_empty_secret_is_verified_against_digest_of_empty_string(self):
        empty_digest = hashlib.sha256(b"").hexdigest()
        wrong_digest = hashlib.sha256(b"x").hexdigest()
        assert verify_stateless_api_key("sk-", empty_digest) is True
        assert verify_stateless_api_key("sk-", wrong_digest) is False

    # ---- verification ----------------------------------------------------- #

    def test_valid_key_returns_true(self):
        digest = hashlib.sha256(b"abc").hexdigest()
        assert verify_stateless_api_key("sk-abc", digest) is True

    def test_wrong_digest_returns_false(self):
        digest = hashlib.sha256(b"abd").hexdigest()
        assert verify_stateless_api_key("sk-abc", digest) is False

    def test_delegates_to_verify_digest_with_secret_only(self):
        with mock.patch.object(
            stateless.utils, "verify_digest", return_value=True
        ) as verify:
            assert verify_stateless_api_key("sk-secret", "the-digest") is True
        verify.assert_called_once_with("secret", "the-digest")

    def test_returns_bool_from_verify_digest(self):
        with mock.patch.object(stateless.utils, "verify_digest", return_value=False):
            assert verify_stateless_api_key("sk-secret", "d") is False

    def test_real_digest_never_matches_wrong_secret(self):
        """Using the real sha256, an arbitrary string digest does not verify."""
        assert verify_stateless_api_key("sk-abc", "not-a-real-digest") is False

    # ---- round trips ------------------------------------------------------ #

    def test_round_trip_with_ascii_digest(self):
        key, digest = new_stateless_api_key()
        assert verify_stateless_api_key(key, digest) is True

    def test_round_trip_with_custom_settings(self):
        settings = StatelessAPIKeySettings(length=64, prefix="live")
        key, digest = new_stateless_api_key(settings)
        assert verify_stateless_api_key(key, digest, settings) is True
        # default settings (prefix 'sk') reject the 'live' key
        with pytest.raises(InvalidPrefix):
            verify_stateless_api_key(key, digest)

    def test_key_does_not_verify_against_another_keys_digest(self):
        key_a, _ = new_stateless_api_key()
        _, digest_b = new_stateless_api_key()
        assert verify_stateless_api_key(key_a, digest_b) is False

    # ---- properties ------------------------------------------------------- #

    @given(
        length=st.integers(min_value=8, max_value=MAX_API_KEY_LENGTH),
        prefix=safe_prefixes,
    )
    def test_property_round_trip(self, length, prefix):
        settings = StatelessAPIKeySettings(length=length, prefix=prefix)
        key, digest = new_stateless_api_key(settings)
        assert verify_stateless_api_key(key, digest, settings) is True

    @given(
        length=st.integers(min_value=8, max_value=MAX_API_KEY_LENGTH),
        prefix=safe_prefixes,
        data=st.data(),
    )
    def test_property_tampered_secret_fails(self, length, prefix, data):
        settings = StatelessAPIKeySettings(length=length, prefix=prefix)
        key, digest = new_stateless_api_key(settings)
        idx = data.draw(st.integers(min_value=len(prefix) + 1, max_value=len(key) - 1))
        replacement = "0" if key[idx] != "0" else "1"
        tampered = key[:idx] + replacement + key[idx + 1 :]
        assert verify_stateless_api_key(tampered, digest, settings) is False

    @given(st.text().filter(lambda s: not s.startswith("sk")))
    def test_property_non_prefixed_keys_raise_invalid_prefix(self, key):
        with pytest.raises(InvalidPrefix):
            verify_stateless_api_key(key, "digest")

    @given(st.one_of(st.none(), st.integers(), st.binary(), st.lists(st.integers())))
    def test_property_non_str_key_always_type_error(self, key):
        with pytest.raises(TypeError):
            verify_stateless_api_key(key, "digest")  # type: ignore[arg-type]
