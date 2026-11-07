"""Cryptographic verification tests for Ed25519 execution grants."""
import base64
import json
import unittest

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
except ImportError:
    Ed25519PrivateKey = None

from kubu_hai.authorization import AuthorizationError, Ed25519GrantVerifier


@unittest.skipIf(Ed25519PrivateKey is None, "install the security extra")
class Ed25519VerifierTests(unittest.TestCase):
    def setUp(self):
        self.private = Ed25519PrivateKey.generate()
        self.public = self.private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.verifier = Ed25519GrantVerifier(self.public)
        self.claims = {
            "sub": "lmlm-agent", "aud": "kubu-hai",
            "task": "text.generate", "nbf": 100, "exp": 200,
            "jti": "nonce-123", "backend": "cpu",
        }

    def _token(self, claims=None):
        payload = base64.urlsafe_b64encode(
            json.dumps(claims or self.claims, separators=(",", ":")).encode()
        ).rstrip(b"=").decode()
        signature = base64.urlsafe_b64encode(
            self.private.sign(payload.encode("ascii"))
        ).rstrip(b"=").decode()
        return payload + "." + signature

    def test_valid_signed_grant_is_parsed(self):
        grant = self.verifier.verify(self._token())
        self.assertEqual(grant.subject, "lmlm-agent")
        self.assertEqual(grant.task, "text.generate")
        self.assertEqual(grant.backend, "cpu")

    def test_tampered_payload_is_rejected(self):
        token = self._token()
        payload, signature = token.split(".")
        tampered = payload[:-1] + ("A" if payload[-1] != "A" else "B")
        with self.assertRaises(AuthorizationError):
            self.verifier.verify(tampered + "." + signature)

    def test_missing_claim_is_rejected(self):
        claims = dict(self.claims)
        del claims["jti"]
        with self.assertRaises(AuthorizationError):
            self.verifier.verify(self._token(claims))

    def test_malformed_token_is_rejected(self):
        with self.assertRaises(AuthorizationError):
            self.verifier.verify("not.a.valid.token")


if __name__ == "__main__":
    unittest.main()
