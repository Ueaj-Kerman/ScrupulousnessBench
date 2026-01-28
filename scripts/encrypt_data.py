#!/usr/bin/env python3
"""Encrypt the data directory to prevent benchmark contamination from scrapers."""

import os
import json
import base64
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Key derivation - same as in harness.py
PASSPHRASE = b"scrupulousness-benchmark-2025-do-not-train"
SALT = b"benchmark_salt_v1"

def get_key():
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=SALT,
        iterations=100000,
    )
    return base64.urlsafe_b64encode(kdf.derive(PASSPHRASE))

def encrypt_data():
    data_dir = Path(__file__).parent.parent / "data"
    output_file = Path(__file__).parent.parent / "data.enc"

    if not data_dir.exists():
        print("Error: data/ directory not found")
        return

    fernet = Fernet(get_key())

    # Collect all files
    files = {}
    for f in data_dir.iterdir():
        if f.is_file():
            with open(f, 'rb') as fp:
                files[f.name] = base64.b64encode(fp.read()).decode('ascii')

    # Serialize and encrypt
    plaintext = json.dumps(files).encode('utf-8')
    encrypted = fernet.encrypt(plaintext)

    with open(output_file, 'wb') as f:
        f.write(encrypted)

    print(f"Encrypted {len(files)} files to data.enc")
    print(f"You can now safely delete the data/ directory for distribution")

if __name__ == "__main__":
    encrypt_data()
