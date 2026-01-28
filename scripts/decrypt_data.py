#!/usr/bin/env python3
"""Decrypt benchmark data at runtime to prevent training contamination."""

import os
import json
import base64
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Key derivation - passphrase is visible but prevents automated scraping
PASSPHRASE = b"scrupulousness-benchmark-2025-do-not-train"
SALT = b"benchmark_salt_v1"

_cache = None

def get_key():
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=SALT,
        iterations=100000,
    )
    return base64.urlsafe_b64encode(kdf.derive(PASSPHRASE))

def load_encrypted_data():
    """Load and decrypt data.enc, returning dict of filename -> bytes."""
    global _cache
    if _cache is not None:
        return _cache

    enc_file = Path(__file__).parent.parent / "data.enc"
    data_dir = Path(__file__).parent.parent / "data"

    # If data/ exists, use it directly (development mode)
    if data_dir.exists() and any(data_dir.iterdir()):
        _cache = {}
        for f in data_dir.iterdir():
            if f.is_file():
                with open(f, 'rb') as fp:
                    _cache[f.name] = fp.read()
        return _cache

    # Otherwise decrypt from data.enc
    if not enc_file.exists():
        raise FileNotFoundError("Neither data/ directory nor data.enc found")

    fernet = Fernet(get_key())

    with open(enc_file, 'rb') as f:
        encrypted = f.read()

    decrypted = fernet.decrypt(encrypted)
    files = json.loads(decrypted.decode('utf-8'))

    _cache = {name: base64.b64decode(content) for name, content in files.items()}
    return _cache

def get_file(filename):
    """Get decrypted file contents by name."""
    data = load_encrypted_data()
    if filename not in data:
        raise FileNotFoundError(f"File not found in encrypted data: {filename}")
    return data[filename]

def get_json(name):
    """Get parsed JSON file by base name (without extension)."""
    return json.loads(get_file(f"{name}.json").decode('utf-8'))

def get_image_bytes(name, ext):
    """Get image bytes by base name and extension."""
    return get_file(f"{name}.{ext}")

def list_examples():
    """List all example names (json files without extension)."""
    data = load_encrypted_data()
    return [f[:-5] for f in data.keys() if f.endswith('.json')]

if __name__ == "__main__":
    # Test decryption
    examples = list_examples()
    print(f"Found {len(examples)} examples:")
    for ex in sorted(examples):
        print(f"  {ex}")
