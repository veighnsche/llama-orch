#!/bin/bash
set -e

echo "🧹 Cleaning up Docker resources..."

# Stop all compose files
docker-compose -f tests/docker/docker-compose.localhost.yml down -v 2>/dev/null || true
docker-compose -f tests/docker/docker-compose.multi-hive.yml down -v 2>/dev/null || true

# Remove images
docker rmi rbee-base:latest rbee-queen:latest rbee-hive:latest 2>/dev/null || true

# Remove dangling images
docker image prune -f

echo "✅ Cleanup complete!"
