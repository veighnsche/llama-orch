#!/bin/bash
set -e

echo "🔧 Building binaries..."
cargo build --bin queen-rbee --bin rbee-hive

echo "🔑 Generating SSH keys..."
./tests/docker/scripts/generate-keys.sh

echo "🐳 Building Docker images..."
docker build -f tests/docker/Dockerfile.base -t rbee-base:latest .
docker build -f tests/docker/Dockerfile.queen -t rbee-queen:latest .
docker build -f tests/docker/Dockerfile.hive -t rbee-hive:latest .

echo "✅ Build complete!"
echo ""
echo "Next steps:"
echo "  1. Start environment: ./tests/docker/scripts/start.sh"
echo "  2. Run tests: ./tests/docker/scripts/test-all.sh"
echo "  3. Stop environment: ./tests/docker/scripts/stop.sh"
