#!/bin/bash
set -e

TOPOLOGY=${1:-localhost}

echo "🐳 Starting Docker environment: $TOPOLOGY"
docker-compose -f tests/docker/docker-compose.$TOPOLOGY.yml up -d

echo "⏳ Waiting for services to be healthy..."
sleep 10

# Wait for queen
echo -n "Checking queen health... "
for i in {1..30}; do
    if curl -sf http://localhost:8500/health > /dev/null 2>&1; then
        echo "✅ Queen ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Queen failed to start"
        docker logs rbee-queen-localhost 2>&1 | tail -20
        exit 1
    fi
    sleep 1
done

# Wait for hive
echo -n "Checking hive health... "
for i in {1..30}; do
    if curl -sf http://localhost:9000/health > /dev/null 2>&1; then
        echo "✅ Hive ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Hive failed to start"
        docker logs rbee-hive-localhost 2>&1 | tail -20
        exit 1
    fi
    sleep 1
done

echo ""
echo "✅ Environment ready!"
echo ""
echo "Services:"
echo "  - Queen: http://localhost:8500"
echo "  - Hive:  http://localhost:9000"
echo "  - SSH:   ssh -i tests/docker/keys/test_id_rsa -p 2222 rbee@localhost"
echo ""
echo "Logs:"
echo "  - docker logs rbee-queen-localhost"
echo "  - docker logs rbee-hive-localhost"
