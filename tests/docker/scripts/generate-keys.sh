#!/bin/bash
set -e

cd "$(dirname "$0")/../keys"

if [ -f "test_id_rsa" ]; then
    echo "⚠️  SSH keys already exist, skipping generation"
    exit 0
fi

ssh-keygen -t ed25519 -f test_id_rsa -N "" -C "rbee-docker-tests"
chmod 600 test_id_rsa
chmod 644 test_id_rsa.pub

echo "✅ SSH keys generated at tests/docker/keys/"
