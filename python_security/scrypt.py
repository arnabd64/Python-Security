import base64
import hashlib
import hmac
import secrets
import unicodedata


class SCryptHasher:
    MAX_PASSWORD_LENGTH: int = 256
    MAX_SALT_LENGTH: int = 128
    MAX_DIGEST_LENGTH: int = 128
    OPENSSL_MAX_MEMORY: int = (2**31) - 1

    def __init__(
        self,
        salt_length: int = 32,
        compute_cost: int = 17,
        block_size: int = 8,
        parallelism: int = 1,
        key_length: int = 32,
    ) -> None:
        self.salt_length = salt_length
        self.ln = compute_cost
        self.r = block_size
        self.p = parallelism
        self.dklen = key_length

        if self.required_memory >= self.OPENSSL_MAX_MEMORY:
            raise ValueError(
                "Required Memory is larger than allowed memory, adjust the parameter values or use default"
            )

    @property
    def required_memory(self) -> int:
        return 128 * self.r * ((1 << self.ln) + self.p + 2) + (1 << 20)

    def _b64_encoder(self, data: bytes) -> str:
        return base64.b64encode(data).rstrip(b"=").decode("ascii")

    def _b64_decoder(self, data: str) -> bytes:
        return base64.b64decode(data + "=" * (-len(data) % 4), validate=True)

    def _phc_encoder(self, salt: bytes, digest: bytes) -> str:
        """
        Converts the salt and digest into a Password Hashing Compatible (PHC)
        string format. Ex: $<id>$<param>=<value>(,<param>=<value>)$<salt>$<hash>

        Args:
            - `salt` (bytes) : Salt in byestarray
            - `digest` (bytes) : Password Digest

        Returns:
            - str : PHC compliant string
        """
        return f"$scrypt$ln={self.ln},r={self.r},p={self.p}${self._b64_encoder(salt)}${self._b64_encoder(digest)}"

    def _phc_decoder(self, phc_string: str) -> dict:
        """
        Converts a Password Hashing Compatible (PHC) string into it's components

        Args:
            - `phc_string` (str) : PHC compliant string

        Returns:
            - dict: dictionary of all the extracted components
        """
        # check if the string is actually a string
        if not isinstance(phc_string, str):
            raise TypeError("input is not a string")

        # extract the components
        empty, algo, params, b64_salt, b64_digest = phc_string.split("$")

        if empty or algo != "scrypt":
            raise ValueError(f"Invalid Hashed Password format: {phc_string}")

        # extract the parameters
        kv: dict[str, int] = {}
        for param in params.split(","):
            key, value = param.split("=")
            kv[key] = int(value)

        parsed = {
            "salt": self._b64_decoder(b64_salt),
            "digest": self._b64_decoder(b64_digest),
            "ln": kv["ln"],
            "r": kv["r"],
            "p": kv["p"],
        }

        # validations
        if len(parsed["salt"]) > self.MAX_SALT_LENGTH:
            raise ValueError("Salt is too Long")
        if len(parsed["digest"]) > self.MAX_DIGEST_LENGTH:
            raise ValueError("Digest is too Long")

        return parsed

    def _encode_password(self, plain_text_password: str) -> bytes:
        password: bytes = unicodedata.normalize("NFKC", plain_text_password).encode(
            "utf-8"
        )
        if len(password) > self.MAX_PASSWORD_LENGTH:
            raise ValueError("Password too Long")
        return password

    def hash_password(self, plain_text_password: str) -> str:
        # generate salt
        salt: bytes = secrets.token_bytes(self.salt_length)

        # hash the password
        digest: bytes = hashlib.scrypt(
            self._encode_password(plain_text_password),
            salt=salt,
            n=(1 << self.ln),
            r=self.r,
            p=self.p,
            dklen=self.dklen,
            maxmem=self.required_memory,
        )

        # convert to PHC string
        phc_string: str = self._phc_encoder(salt, digest)

        return phc_string

    def verify_password(self, plain_text_password: str, hashed_password: str) -> bool:
        # parse the hashed password
        parsed_password: dict = self._phc_decoder(hashed_password)

        # hash the plain text password
        recieved: bytes = hashlib.scrypt(
            self._encode_password(plain_text_password),
            salt=parsed_password["salt"],
            n=(1 << parsed_password["ln"]),
            r=parsed_password["r"],
            p=parsed_password["p"],
            dklen=len(parsed_password["digest"]),
            maxmem=self.required_memory,
        )

        return hmac.compare_digest(recieved, parsed_password["digest"])

    def needs_rehash(self, hashed_password: str) -> bool:
        parsed_password: dict = self._phc_decoder(hashed_password)

        if parsed_password["ln"] != self.ln:
            return True
        if parsed_password["r"] != self.r:
            return True
        if parsed_password["p"] != self.p:
            return True
        if len(parsed_password["salt"]) != self.salt_length:
            return True
        if len(parsed_password["digest"]) != self.dklen:
            return True
        return False
