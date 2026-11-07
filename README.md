# Python Security

Small, dependency-free security utilities for Python applications.

`python-security` provides simple building blocks for common authentication tasks:

- Password hashing and verification with **scrypt**.
- JSON Web Token (JWT) creation and verification using **HS256**.
- Random API key generation and verification.
- Time-based one-time passwords (**TOTP**) for two-factor authentication (2FA).

The library uses Python's standard library and has no third-party runtime dependencies. It is intended for developers who want to add these basic features to their own applications without bringing in a separate authentication framework.

> **Project status:** This project is still evolving. The package provides primitives, not a complete authentication system. Review the security notes and current limitations before using it in a production application.

## Requirements

- Python **3.12** or later.
- No third-party runtime packages are required.

## Installation

### Install directly from GitHub

The package can be installed from this repository:

```bash
python -m pip install "git+https://github.com/arnabd64/Python-Security.git"
```

### Install from a local clone

```bash
git clone https://github.com/arnabd64/Python-Security.git
cd Python-Security
python -m pip install .
```

To install an editable copy while developing the package:

```bash
python -m pip install -e .
```

The repository also defines development tools in its `dev` dependency group. With [uv](https://docs.astral.sh/uv/), install those tools and run the test suite with:

```bash
uv sync --group dev
uv run pytest
```

## Quick start

The examples below show the main functions. Each feature can be used independently.

### 1. Password hashing

Use `hash_password()` when a user creates or changes a password. Store the returned hash in your database instead of storing the original password.

```python
from python_security.scrypt import hash_password, verify_password

# During registration or a password change
password_hash = hash_password("correct-horse-battery-staple")

# Store password_hash in your database.

# During login, load the saved hash and verify the submitted password.
if verify_password("correct-horse-battery-staple", password_hash):
    print("Password is correct")
else:
    print("Invalid username or password")
```

The default configuration uses `hashlib.scrypt` with a fresh random 32-byte salt for every password. The result is stored in a PHC-style string that includes the algorithm parameters, salt, and derived hash. This means the parameters needed for verification are stored with the hash.

The default maximum password length is 256 characters. You can provide your own settings through `SCryptSettings`:

```python
from python_security.scrypt import hash_password, verify_password, SCryptSettings

# init SCrypt with your own settings
settings = SCryptSettings(
    ln=17,
    r=8,
    p=1,
    dklen=32,
    salt_length=32,
    max_password_length=256,
)

# or use our own default settings
settings = SCryptSettings.default()

# hash a password
password_hash: str = hash_password("your-password", settings=settings)

# verify it
password_test: bool = verify_password("your-password", password_hash)
```

### 2. JWT access and refresh tokens

The JWT module supports signed access and refresh tokens using **HMAC-SHA256 (HS256)**.

First, set a secret key. It must be between 32 and 64 characters and should be generated randomly. For example, this command creates a 32-byte random value represented as 64 hexadecimal characters:

```bash
export JWT_SECRET="$(python -c 'import secrets; print(secrets.token_hex(32))')"
```

You can optionally configure the token lifetimes, issuer, and audience:

```bash
export JWT_ACCESS_TTL=900
export JWT_REFRESH_TTL=432000
export JWT_ISSUER="your-application"
export JWT_AUDIENCE="your-api"
```

The defaults are 900 seconds (15 minutes) for access tokens and 432,000 seconds (5 days) for refresh tokens. The default issuer is `JWTService` and the default audience is `All`.

Create and verify tokens:

```python
from python_security.jwt import (
    issue_access_token,
    issue_refresh_token,
    parse_token,
    reissue_access_token,
    verify_token,
)
from python_security.jwt.config import JWTSettings
from python_security.exceptions import JWTException

# use your own settings
settings = JWTSettings.from_env()

# or use our default settings (only requires JWT_SECRET)
settings = JWTSettings.default()

# Create an access token. The function returns (token, jti).
access_token, access_jti = issue_access_token(
    sub="user-123",
    claims={"role": "user"},
    settings=settings,
)

# Create a refresh token when the user signs in.
refresh_token, refresh_jti = issue_refresh_token(
    sub="user-123",
    settings=settings,
)

# Parse and validate an access token.
try:
    payload = parse_token(access_token, settings=settings)
    print(payload.sub)  # user-123
    print(payload.claims)  # {'role': 'user'}
except JWTException:
    # Reject the request if the token is invalid or expired.
    pass

# Or use verify_token() when you only need a True/False result.
if verify_token(access_token, settings=settings):
    print("Token is valid")

# Exchange a valid refresh token for a new access token.
new_access_token, new_access_jti = reissue_access_token(
    refresh_token,
    settings=settings,
)
```

`parse_token()` returns a `Payload` object. Its useful fields include `sub`, `claims`, `jti`, `iat`, `exp`, `iss`, `aud`, and `token_type`.

**Important:** JWT payloads are encoded, not encrypted. Do not put passwords, API keys, or other secrets in token claims. The library does not maintain a token blacklist or session database, so immediate logout, revocation, refresh-token rotation, and reuse detection must be implemented by your application.

### 3. API key generation and verification

Use API keys for machine-to-machine access or integrations. Generate a key once, show the plain key to the client once, and store the returned digest in your database.

```python
from python_security.api_keys import (
    new_stateless_api_key,
    verify_stateless_api_key,
)

# Generate a new key.
api_key, api_key_digest = new_stateless_api_key()

# Show api_key to the client once. Store api_key_digest with the
# account or integration that owns the key; do not store the plain key.

# Later, load the saved digest and verify a key supplied by the client.
if verify_stateless_api_key(supplied_api_key, api_key_digest):
    print("API key is valid")
else:
    print("Invalid API key")
```

By default, keys use the `sk-` prefix and a 32-byte random secret (64 hexadecimal characters). The digest is a SHA-256 hash of the random secret portion of the key. SHA-256 is suitable here because the key is randomly generated and has high entropy; it should not be used as a password-hashing function.

You can configure the key length and prefix:

```python
from python_security.api_keys import StatelessAPIKeySettings, new_stateless_api_key

settings = StatelessAPIKeySettings(length=32, prefix="myapp")
api_key, api_key_digest = new_stateless_api_key(settings)
```

Use a prefix without a hyphen. The current verifier expects the generated key to contain one hyphen separating the prefix from the secret.

**Important:** These keys are stateless. The library does not store keys, associate them with users, set expiry dates, or revoke them. Store each digest with the relevant owner and implement expiry, disablement, and rotation in your application.

### 4. TOTP two-factor authentication

The TOTP module provides the core functions needed to add an authenticator-app step to a login flow:

1. Generate a secret for the user.
2. Create an `otpauth://` provisioning URI.
3. Let the user register that secret in an authenticator app.
4. Verify the code entered by the user during enrollment and login.

```python
from python_security.totp.core import (
    new_totp_secret,
    totp_uri,
    verify_otp,
)

# Generate this once when the user starts setting up 2FA.
secret = new_totp_secret()

# Give the user a provisioning URI for their authenticator app.
provisioning_uri = totp_uri(
    secret,
    account_name="user@example.com",
    issuer="My Application",
)

# Display the URI as a QR code in your application, or offer a secure
# manual setup option. The library returns the URI; it does not create
# a QR-code image.

# Store the secret securely so it can be used to verify future codes.
# After the user submits the code from their authenticator app:
if verify_otp(submitted_code, secret):
    print("Authenticator code is valid")
else:
    print("Invalid authenticator code")
```

The default TOTP settings are:

| Setting | Default |
| --- | --- |
| Secret size | 20 bytes |
| Hash algorithm | SHA-1 |
| Code length | 6 digits |
| Time step | 30 seconds |
| Verification window | 1 step before or after the current step |

You can customize these settings with `TOTPSettings`:

```python
from python_security.totp.config import TOTPSettings
from python_security.totp.core import new_totp_secret, generate_otp, verify_otp

settings = TOTPSettings(
    num_bytes=20,
    algorithm="SHA1",
    digits=6,
    interval=30,
    window=1,
)

secret = new_totp_secret(settings)
current_code = generate_otp(secret, settings=settings)

# When the user submits a code:
is_valid = verify_otp(submitted_code, secret, settings=settings)
```

The TOTP module implements code generation and verification. It does not provide QR-code rendering, recovery codes, account enrollment storage, or a complete login workflow. Store TOTP secrets securely (they must be available again for verification), and require a valid code before enabling 2FA on an account.

## Imports at a glance

| Feature | Import |
| --- | --- |
| Password hashing | `from python_security import hash_password, verify_password` |
| JWT | `from python_security.jwt import issue_access_token, issue_refresh_token, parse_token, reissue_access_token, verify_token` |
| API keys | `from python_security.api_keys import new_stateless_api_key, verify_stateless_api_key` |
| TOTP | `from python_security.totp.core import new_totp_secret, totp_uri, generate_otp, verify_otp` |

## Security notes and current limitations

This package is a collection of primitives. Your application is still responsible for the surrounding authentication workflow and operational controls.

- **Keep secrets out of source control.** Load JWT signing keys and other secrets from environment variables or a secrets manager.
- **Use HTTPS.** Never send passwords, bearer tokens, API keys, TOTP secrets, or provisioning URIs over an untrusted connection.
- **Protect stored data.** Store password hashes rather than passwords. Store API-key digests rather than plain API keys. Encrypt TOTP secrets at rest because they must be recovered to verify codes.
- **Add application-level rate limits.** Protect password, API-key, and OTP verification endpoints against brute-force attempts.
- **JWT revocation is your responsibility.** A valid JWT remains valid until its expiry unless your application adds a denylist or another server-side session/revocation mechanism.
- **API-key lifecycle is your responsibility.** The API-key module does not provide expiry, revocation, rotation, or database storage.
- **TOTP state is process-local.** The current replay-prevention and failed-attempt tracking uses in-memory state. It is not shared between multiple workers or application instances and is cleared when the process restarts. The current failed-attempt counter can also block a secret after 20 failed checks without an automatic time-based reset. Review or replace this behavior before exposing OTP verification to untrusted clients.
- **TOTP secrets need careful handling.** Do not log them or expose provisioning URIs after enrollment. The library does not provide recovery codes or QR-code generation.
- **Use short-lived access tokens.** Keep access-token lifetimes short and treat refresh tokens as especially sensitive credentials.

Do not treat this package as a replacement for a full identity provider or a security review. Test your integration, threat model your application, and review the implementation before using it for sensitive workloads.

## Development

The project uses `pytest`, `hypothesis`, and `pytest-cov` for tests, and `ruff` for linting. Install the development dependency group with `uv sync --group dev`.

Run tests:

```bash
uv run pytest
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
