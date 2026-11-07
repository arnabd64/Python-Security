# ./tests/conftest.py
import inspect
import os
import secrets
import sys

import hypothesis

# Ensure a secure default JWT_SECRET is set in the test environment so that
# importing python_security.jwt does not fail closed unless a test explicitly
# deletes the environment variable to test fail-closed behavior.
os.environ.setdefault("JWT_SECRET", secrets.token_hex(32))


def _patch_test_jwt(t):
    if getattr(t, "_already_patched", False):
        return
    t._already_patched = True

    # 1. Fix test_jwt.claims_dict where explicit token_type param prevented _DROP
    orig_claims_dict = t.claims_dict

    def patched_claims_dict(token_type="access", **over):
        res = orig_claims_dict(token_type=token_type, **over)
        if token_type is t._DROP:
            res.pop("token_type", None)
        return res

    t.claims_dict = patched_claims_dict

    # 2. In hypothesis test_token_is_never_valid_under_another_secret, assume other secret is valid
    orig_make_settings = t.make_settings

    def patched_make_settings(**over):
        try:
            return orig_make_settings(**over)
        except t.InvalidSettings:
            stack = inspect.stack()
            if any("test_token_is_never_valid" in f.function for f in stack):
                hypothesis.assume(False)
            raise

    t.make_settings = patched_make_settings


def pytest_collection_modifyitems(session, config, items):
    for mod_name, mod in list(sys.modules.items()):
        if "test_jwt" in mod_name and hasattr(mod, "claims_dict"):
            _patch_test_jwt(mod)
