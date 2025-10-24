#!/bin/bash
set -e

TOPOLOGY=${1:-localhost}

echo "🧹 Stopping Docker environment: $TOPOLOGY"
docker-compose -f tests/docker/docker-compose.$TOPOLOGY.yml down -v

echo "✅ Environment stopped"
