"""Regression checks for deterministic document identifier hashing."""

import hashlib
from uuid import UUID

from paperqa.utils import encode_id


def test_document_ids_use_sha256_and_preserve_normalization():
    expected = hashlib.sha256(b"10.1000/example").hexdigest()
    assert encode_id("10.1000/Example", maxsize=None) == expected
    assert encode_id("10.1000/EXAMPLE") == expected[:16]
    assert encode_id(b"10.1000/example", maxsize=12) == expected[:12]


def test_uuid_identifier_is_deterministic():
    identifier = UUID("12345678-1234-5678-1234-567812345678")
    assert encode_id(identifier) == encode_id(str(identifier))
