"""
Pytest suite for scrypt.SCryptHasher.

Place next to scrypt.py (python_testing/test_scrypt.py) and run:

    pytest python_testing/test_scrypt.py --cov=scrypt --cov-report=term-missing

If your import path differs, change the import below, e.g.
    from python_testing.scrypt import SCryptHasher
"""

import base64
import binascii
import hashlib
import unicodedata

import pytest

from python_security import scrypt
from python_security.scrypt import SCryptHasher

# Low cost parameters keep the suite fast (~1 MB per hash instead of ~128 MB).
FAST_LN = 4
PASSWORD = "correct horse battery staple"


def make_hasher(**overrides) -> SCryptHasher:
    params = {"compute_cost": FAST_LN}
    params.update(overrides)
    return SCryptHasher(**params)


@pytest.fixture
def hasher() -> SCryptHasher:
    return make_hasher()


def b64(data: bytes) -> str:
    """Independent unpadded base64 helper (does not use the class under test)."""
    return base64.b64encode(data).rstrip(b"=").decode("ascii")


# --------------------------------------------------------------------------- #
# __init__ and required_memory
# --------------------------------------------------------------------------- #
class TestInit:
    def test_default_parameters(self):
        h = SCryptHasher()
        assert h.salt_length == 32
        assert h.ln == 17
        assert h.r == 8
        assert h.p == 1
        assert h.dklen == 32

    def test_custom_parameters_are_stored(self):
        h = SCryptHasher(
            salt_length=16, compute_cost=10, block_size=4, parallelism=2, key_length=64
        )
        assert (h.salt_length, h.ln, h.r, h.p, h.dklen) == (16, 10, 4, 2, 64)

    def test_class_limits(self):
        assert SCryptHasher.MAX_PASSWORD_LENGTH == 256
        assert SCryptHasher.MAX_SALT_LENGTH == 128
        assert SCryptHasher.MAX_DIGEST_LENGTH == 128
        assert SCryptHasher.OPENSSL_MAX_MEMORY == 2**31 - 1

    def test_highest_allowed_cost_is_accepted(self):
        # ln=20, r=8 -> ~1.07 GB, below the 2 GiB limit
        assert (
            make_hasher(compute_cost=20).required_memory
            < SCryptHasher.OPENSSL_MAX_MEMORY
        )

    def test_memory_over_limit_raises(self):
        # ln=21, r=8 -> just over 2 GiB
        with pytest.raises(ValueError, match="Required Memory is larger"):
            SCryptHasher(compute_cost=21)

    def test_large_block_size_raises(self):
        with pytest.raises(ValueError, match="Required Memory is larger"):
            SCryptHasher(compute_cost=17, block_size=200)


class TestRequiredMemory:
    @pytest.mark.parametrize(
        "ln, r, p",
        [(4, 8, 1), (10, 8, 1), (17, 8, 1), (12, 4, 2), (14, 16, 3)],
    )
    def test_formula(self, ln, r, p):
        h = SCryptHasher(compute_cost=ln, block_size=r, parallelism=p)
        assert h.required_memory == 128 * r * ((1 << ln) + p + 2) + (1 << 20)

    def test_default_is_about_135_mb(self):
        assert SCryptHasher().required_memory == 128 * 8 * (2**17 + 3) + 2**20


# --------------------------------------------------------------------------- #
# _b64_encoder / _b64_decoder
# --------------------------------------------------------------------------- #
class TestBase64Helpers:
    @pytest.mark.parametrize(
        "data", [b"", b"A", b"AB", b"ABC", b"ABCD", bytes(range(256)), b"\xff" * 33]
    )
    def test_roundtrip(self, hasher, data):
        assert hasher._b64_decoder(hasher._b64_encoder(data)) == data

    def test_encoder_strips_padding(self, hasher):
        assert hasher._b64_encoder(b"A") == "QQ"
        assert "=" not in hasher._b64_encoder(b"AB")

    def test_encoder_returns_ascii_str(self, hasher):
        out = hasher._b64_encoder(b"\x00\x01\x02")
        assert isinstance(out, str)
        out.encode("ascii")

    def test_decoder_restores_missing_padding(self, hasher):
        assert hasher._b64_decoder("QQ") == b"A"
        assert hasher._b64_decoder("QUI") == b"AB"
        assert hasher._b64_decoder("QUJD") == b"ABC"  # no padding needed

    def test_decoder_rejects_invalid_characters(self, hasher):
        with pytest.raises(binascii.Error):
            hasher._b64_decoder("@@@@")

    def test_decoder_rejects_impossible_length(self, hasher):
        # 5 chars -> remainder 1 is never valid base64
        with pytest.raises(binascii.Error):
            hasher._b64_decoder("QUJDR")


# --------------------------------------------------------------------------- #
# _phc_encoder
# --------------------------------------------------------------------------- #
class TestPhcEncoder:
    def test_exact_format(self):
        h = SCryptHasher(compute_cost=4, block_size=8, parallelism=1)
        salt, digest = b"saltsalt", b"digestdigest"
        assert (
            h._phc_encoder(salt, digest)
            == f"$scrypt$ln=4,r=8,p=1${b64(salt)}${b64(digest)}"
        )

    def test_uses_instance_parameters(self):
        h = SCryptHasher(compute_cost=6, block_size=4, parallelism=3)
        assert h._phc_encoder(b"s", b"d").startswith("$scrypt$ln=6,r=4,p=3$")

    def test_has_five_dollar_separated_fields(self, hasher):
        parts = hasher._phc_encoder(b"s", b"d").split("$")
        assert len(parts) == 5 and parts[0] == "" and parts[1] == "scrypt"


# --------------------------------------------------------------------------- #
# _phc_decoder
# --------------------------------------------------------------------------- #
class TestPhcDecoder:
    def test_roundtrip_with_encoder(self):
        h = SCryptHasher(compute_cost=5, block_size=4, parallelism=2)
        salt, digest = bytes(range(16)), bytes(range(32))
        parsed = h._phc_decoder(h._phc_encoder(salt, digest))
        assert parsed == {"salt": salt, "digest": digest, "ln": 5, "r": 4, "p": 2}

    @pytest.mark.parametrize(
        "bad", [None, 123, b"$scrypt$ln=4,r=8,p=1$AAAA$AAAA", ["x"]]
    )
    def test_non_string_raises_type_error(self, hasher, bad):
        with pytest.raises(TypeError, match="not a string"):
            hasher._phc_decoder(bad)

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "garbage",
            "a$b",
            "$scrypt$ln=4,r=8,p=1$AAAA",  # too few fields
            "$scrypt$ln=4,r=8,p=1$AAAA$AAAA$extra",  # too many fields
        ],
    )
    def test_wrong_field_count_raises_value_error(self, hasher, bad):
        with pytest.raises(ValueError):
            hasher._phc_decoder(bad)

    def test_leading_text_before_first_dollar_raises(self, hasher):
        with pytest.raises(ValueError, match="Invalid Hashed Password format"):
            hasher._phc_decoder("x$scrypt$ln=4,r=8,p=1$AAAA$AAAA")

    @pytest.mark.parametrize("algo", ["bcrypt", "argon2id", "SCRYPT", ""])
    def test_wrong_algorithm_raises(self, hasher, algo):
        with pytest.raises(ValueError, match="Invalid Hashed Password format"):
            hasher._phc_decoder(f"${algo}$ln=4,r=8,p=1$AAAA$AAAA")

    def test_missing_parameter_raises_key_error(self, hasher):
        with pytest.raises(KeyError):
            hasher._phc_decoder("$scrypt$r=8,p=1$AAAA$AAAA")  # no ln

    def test_non_integer_parameter_raises_value_error(self, hasher):
        with pytest.raises(ValueError):
            hasher._phc_decoder("$scrypt$ln=abc,r=8,p=1$AAAA$AAAA")

    def test_parameter_without_equals_raises_value_error(self, hasher):
        with pytest.raises(ValueError):
            hasher._phc_decoder("$scrypt$ln=4,r=8,p$AAAA$AAAA")

    def test_invalid_base64_salt_raises(self, hasher):
        with pytest.raises(binascii.Error):
            hasher._phc_decoder("$scrypt$ln=4,r=8,p=1$@@@@$AAAA")

    def test_invalid_base64_digest_raises(self, hasher):
        with pytest.raises(binascii.Error):
            hasher._phc_decoder("$scrypt$ln=4,r=8,p=1$AAAA$@@@@")

    def test_salt_at_limit_is_accepted(self, hasher):
        salt = b"\x01" * SCryptHasher.MAX_SALT_LENGTH
        parsed = hasher._phc_decoder(hasher._phc_encoder(salt, b"d" * 32))
        assert parsed["salt"] == salt

    def test_salt_over_limit_raises(self, hasher):
        salt = b"\x01" * (SCryptHasher.MAX_SALT_LENGTH + 1)
        with pytest.raises(ValueError, match="Salt is too Long"):
            hasher._phc_decoder(hasher._phc_encoder(salt, b"d" * 32))

    def test_digest_at_limit_is_accepted(self, hasher):
        digest = b"\x02" * SCryptHasher.MAX_DIGEST_LENGTH
        parsed = hasher._phc_decoder(hasher._phc_encoder(b"s" * 16, digest))
        assert parsed["digest"] == digest

    def test_digest_over_limit_raises(self, hasher):
        digest = b"\x02" * (SCryptHasher.MAX_DIGEST_LENGTH + 1)
        with pytest.raises(ValueError, match="Digest is too Long"):
            hasher._phc_decoder(hasher._phc_encoder(b"s" * 16, digest))


# --------------------------------------------------------------------------- #
# _encode_password
# --------------------------------------------------------------------------- #
class TestEncodePassword:
    def test_ascii_returns_utf8_bytes(self, hasher):
        assert hasher._encode_password("hello") == b"hello"

    def test_empty_password_is_encodable(self, hasher):
        assert hasher._encode_password("") == b""

    def test_nfkc_normalisation_applied(self, hasher):
        # U+FB01 (fi ligature) -> "fi" under NFKC
        assert hasher._encode_password("\ufb01") == b"fi"

    def test_composed_and_decomposed_forms_match(self, hasher):
        composed = "\u00e9"  # é
        decomposed = "e\u0301"
        assert hasher._encode_password(composed) == hasher._encode_password(decomposed)

    def test_fullwidth_characters_normalised(self, hasher):
        assert hasher._encode_password("\uff21\uff22") == b"AB"

    def test_unicode_encoded_as_utf8(self, hasher):
        pw = "pässwörd🔑"
        expected = unicodedata.normalize("NFKC", pw).encode("utf-8")
        assert hasher._encode_password(pw) == expected

    def test_exactly_max_length_accepted(self, hasher):
        pw = "a" * SCryptHasher.MAX_PASSWORD_LENGTH
        assert len(hasher._encode_password(pw)) == 256

    def test_over_max_length_rejected(self, hasher):
        with pytest.raises(ValueError, match="Password too Long"):
            hasher._encode_password("a" * (SCryptHasher.MAX_PASSWORD_LENGTH + 1))

    def test_limit_is_measured_in_bytes_not_characters(self, hasher):
        # 100 emoji = 100 characters but 400 UTF-8 bytes
        with pytest.raises(ValueError, match="Password too Long"):
            hasher._encode_password("🔑" * 100)

    def test_length_checked_after_normalisation(self, hasher):
        # 200 ligatures (3 bytes each = 600) normalise to 400 bytes -> still too long,
        # but 100 ligatures normalise to 200 bytes -> accepted although raw is 300 bytes.
        assert len(hasher._encode_password("\ufb01" * 100)) == 200
        with pytest.raises(ValueError):
            hasher._encode_password("\ufb01" * 200)


# --------------------------------------------------------------------------- #
# hash_password
# --------------------------------------------------------------------------- #
class TestHashPassword:
    def test_returns_phc_string(self, hasher):
        out = hasher.hash_password(PASSWORD)
        assert isinstance(out, str)
        assert out.startswith(f"$scrypt$ln={FAST_LN},r=8,p=1$")
        assert len(out.split("$")) == 5

    def test_password_not_in_output(self, hasher):
        assert PASSWORD not in hasher.hash_password(PASSWORD)

    def test_salt_and_digest_lengths(self):
        h = make_hasher(salt_length=24, key_length=48)
        parsed = h._phc_decoder(h.hash_password(PASSWORD))
        assert len(parsed["salt"]) == 24
        assert len(parsed["digest"]) == 48

    def test_same_password_gives_different_hashes(self, hasher):
        assert hasher.hash_password(PASSWORD) != hasher.hash_password(PASSWORD)

    def test_salts_are_unique_across_many_hashes(self, hasher):
        salts = {
            hasher._phc_decoder(hasher.hash_password("x"))["salt"] for _ in range(20)
        }
        assert len(salts) == 20

    def test_matches_independent_scrypt_computation(self, hasher, monkeypatch):
        """Known-answer test: output must equal a direct hashlib.scrypt call."""
        fixed_salt = bytes(range(32))
        monkeypatch.setattr("python_security.scrypt.secrets.token_bytes", lambda n: fixed_salt[:n])

        expected_digest = hashlib.scrypt(
            PASSWORD.encode(), salt=fixed_salt, n=1 << FAST_LN, r=8, p=1, dklen=32
        )
        assert hasher.hash_password(PASSWORD) == (
            f"$scrypt$ln={FAST_LN},r=8,p=1${b64(fixed_salt)}${b64(expected_digest)}"
        )

    def test_uses_secrets_for_salt(self, hasher, monkeypatch):
        calls = []

        def fake_token_bytes(n):
            calls.append(n)
            return b"\x07" * n

        monkeypatch.setattr("python_security.scrypt.secrets.token_bytes", fake_token_bytes)
        hasher.hash_password(PASSWORD)
        assert calls == [hasher.salt_length]

    def test_empty_password_can_be_hashed(self, hasher):
        assert hasher.verify_password("", hasher.hash_password(""))

    def test_overlong_password_rejected(self, hasher):
        with pytest.raises(ValueError, match="Password too Long"):
            hasher.hash_password("a" * 257)

    def test_non_string_password_rejected(self, hasher):
        with pytest.raises(TypeError):
            hasher.hash_password(b"bytes-not-str")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"block_size": 4},
            {"parallelism": 2},
            {"key_length": 64},
            {"salt_length": 16},
        ],
    )
    def test_custom_parameters_roundtrip(self, kwargs):
        h = make_hasher(**kwargs)
        assert h.verify_password(PASSWORD, h.hash_password(PASSWORD))

    def test_default_parameters_end_to_end(self):
        """The single test that uses the production cost factor (ln=17, ~128 MB)."""
        h = SCryptHasher()
        hashed = h.hash_password(PASSWORD)
        assert hashed.startswith("$scrypt$ln=17,r=8,p=1$")
        assert h.verify_password(PASSWORD, hashed)
        assert not h.verify_password(PASSWORD + "x", hashed)
        assert not h.needs_rehash(hashed)


# --------------------------------------------------------------------------- #
# verify_password
# --------------------------------------------------------------------------- #
class TestVerifyPassword:
    def test_correct_password(self, hasher):
        assert hasher.verify_password(PASSWORD, hasher.hash_password(PASSWORD)) is True

    def test_wrong_password(self, hasher):
        assert hasher.verify_password("wrong", hasher.hash_password(PASSWORD)) is False

    @pytest.mark.parametrize(
        "attempt", [PASSWORD.upper(), PASSWORD + " ", " " + PASSWORD, PASSWORD[:-1], ""]
    )
    def test_near_miss_passwords_rejected(self, hasher, attempt):
        assert hasher.verify_password(attempt, hasher.hash_password(PASSWORD)) is False

    def test_unicode_equivalent_forms_verify(self, hasher):
        hashed = hasher.hash_password("caf\u00e9")  # composed
        assert hasher.verify_password("cafe\u0301", hashed)  # decomposed

    def test_independently_built_hash_verifies(self, hasher):
        """Verifies a PHC string created without the class under test."""
        salt = b"0123456789abcdef"
        digest = hashlib.scrypt(PASSWORD.encode(), salt=salt, n=16, r=8, p=1, dklen=32)
        phc = f"$scrypt$ln=4,r=8,p=1${b64(salt)}${b64(digest)}"
        assert hasher.verify_password(PASSWORD, phc)

    def test_tampered_digest_rejected(self, hasher):
        parsed = hasher._phc_decoder(hasher.hash_password(PASSWORD))
        flipped = bytes([parsed["digest"][0] ^ 0x01]) + parsed["digest"][1:]
        tampered = hasher._phc_encoder(parsed["salt"], flipped)
        assert hasher.verify_password(PASSWORD, tampered) is False

    def test_tampered_salt_rejected(self, hasher):
        parsed = hasher._phc_decoder(hasher.hash_password(PASSWORD))
        flipped = bytes([parsed["salt"][0] ^ 0x01]) + parsed["salt"][1:]
        tampered = hasher._phc_encoder(flipped, parsed["digest"])
        assert hasher.verify_password(PASSWORD, tampered) is False

    def test_stored_parameters_are_used_not_instance_parameters(self):
        """A hash made with r=4, p=2 must verify on an instance configured with r=8, p=1."""
        old = make_hasher(block_size=4, parallelism=2, key_length=16, salt_length=8)
        new = make_hasher()
        hashed = old.hash_password(PASSWORD)
        assert new.verify_password(PASSWORD, hashed) is True
        assert new.verify_password("nope", hashed) is False

    def test_digest_length_taken_from_stored_hash(self):
        old = make_hasher(key_length=64)
        hashed = old.hash_password(PASSWORD)
        assert make_hasher(key_length=32).verify_password(PASSWORD, hashed)

    def test_uses_constant_time_comparison(self, hasher, monkeypatch):
        calls = []
        real = __import__("hmac").compare_digest

        def spy(a, b):
            calls.append((a, b))
            return real(a, b)

        monkeypatch.setattr("python_security.scrypt.hmac.compare_digest", spy)
        hasher.verify_password(PASSWORD, hasher.hash_password(PASSWORD))
        assert len(calls) == 1

    def test_hash_from_lower_cost_instance_still_verifies(self):
        """Upgrading cost factor: old (cheaper) hashes must verify on the new instance."""
        old = make_hasher(compute_cost=4)
        new = make_hasher(compute_cost=8)
        assert new.verify_password(PASSWORD, old.hash_password(PASSWORD))

    @pytest.mark.xfail(
        strict=True,
        raises=ValueError,
        reason=(
            "BUG: verify_password passes maxmem=self.required_memory (the *instance's* "
            "memory budget) to scrypt instead of one derived from the *stored* hash's "
            "parameters. A hash created with a higher cost than the verifying instance "
            "fails with 'memory limit exceeded' instead of returning True/False. "
            "Remove this xfail once fixed."
        ),
    )
    def test_hash_from_higher_cost_instance_verifies(self):
        strong = make_hasher(compute_cost=14)
        weak = make_hasher(compute_cost=4)
        assert weak.verify_password(PASSWORD, strong.hash_password(PASSWORD))

    @pytest.mark.parametrize(
        "bad_hash",
        [
            "",
            "not-a-hash",
            "$bcrypt$ln=4,r=8,p=1$AAAA$AAAA",
            "$scrypt$ln=4,r=8,p=1$AAAA",
        ],
    )
    def test_malformed_hash_raises(self, hasher, bad_hash):
        with pytest.raises(ValueError):
            hasher.verify_password(PASSWORD, bad_hash)

    def test_non_string_hash_raises_type_error(self, hasher):
        with pytest.raises(TypeError):
            hasher.verify_password(PASSWORD, None)  # type: ignore[arg-type]

    def test_overlong_password_raises(self, hasher):
        hashed = hasher.hash_password(PASSWORD)
        with pytest.raises(ValueError, match="Password too Long"):
            hasher.verify_password("a" * 257, hashed)


# --------------------------------------------------------------------------- #
# needs_rehash
# --------------------------------------------------------------------------- #
class TestNeedsRehash:
    def test_identical_parameters_do_not_need_rehash(self, hasher):
        assert hasher.needs_rehash(hasher.hash_password(PASSWORD)) is False

    @pytest.mark.parametrize(
        "old_kwargs",
        [
            {"compute_cost": FAST_LN + 1},
            {"block_size": 4},
            {"parallelism": 2},
            {"salt_length": 16},
            {"key_length": 64},
        ],
        ids=["ln", "r", "p", "salt_length", "key_length"],
    )
    def test_each_parameter_mismatch_triggers_rehash(self, old_kwargs):
        old_hash = make_hasher(**old_kwargs).hash_password(PASSWORD)
        assert make_hasher().needs_rehash(old_hash) is True

    def test_lower_cost_than_current_triggers_rehash(self):
        old_hash = make_hasher(compute_cost=4).hash_password(PASSWORD)
        assert make_hasher(compute_cost=6).needs_rehash(old_hash) is True

    def test_rehash_flow_produces_up_to_date_hash(self):
        old_hash = make_hasher(compute_cost=4).hash_password(PASSWORD)
        new = make_hasher(compute_cost=6)
        assert new.needs_rehash(old_hash)
        assert new.verify_password(PASSWORD, old_hash)
        upgraded = new.hash_password(PASSWORD)
        assert not new.needs_rehash(upgraded)
        assert new.verify_password(PASSWORD, upgraded)

    def test_malformed_hash_raises(self, hasher):
        with pytest.raises(ValueError):
            hasher.needs_rehash("garbage")

    def test_non_string_raises_type_error(self, hasher):
        with pytest.raises(TypeError):
            hasher.needs_rehash(12345)  # type: ignore[arg-type]
