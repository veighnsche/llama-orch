#!/usr/bin/env bash
# Integration Test Runner with Docker Compose
# Created by: TEAM-106
# Usage: ./run-integration-tests.sh [--build] [--down] [--tags @tag]

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default options
BUILD=false
DOWN=false
TAGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --build)
            BUILD=true
            shift
            ;;
        --down)
            DOWN=true
            shift
            ;;
        --tags)
            TAGS="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║         Integration Test Runner - Docker Compose              ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Cleanup function
cleanup() {
    echo -e "${YELLOW}🧹 Cleaning up...${NC}"
    cd "$SCRIPT_DIR"
    docker-compose -f docker-compose.integration.yml down -v
}

# Handle shutdown
if [[ "$DOWN" == "true" ]]; then
    cleanup
    echo -e "${GREEN}✅ Services stopped${NC}"
    exit 0
fi

# Build if requested
if [[ "$BUILD" == "true" ]]; then
    echo -e "${BLUE}🔨 Building Docker images...${NC}"
    cd "$SCRIPT_DIR"
    docker-compose -f docker-compose.integration.yml build
    echo -e "${GREEN}✅ Build complete${NC}"
    echo ""
fi

# Start services
echo -e "${BLUE}🚀 Starting services...${NC}"
cd "$SCRIPT_DIR"
docker-compose -f docker-compose.integration.yml up -d

# Wait for services to be healthy
echo -e "${YELLOW}⏳ Waiting for services to be healthy...${NC}"
sleep 5

# Check queen-rbee
echo -n "  Checking queen-rbee... "
if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅${NC}"
else
    echo -e "${RED}❌${NC}"
    echo -e "${RED}queen-rbee not healthy, check logs:${NC}"
    docker-compose -f docker-compose.integration.yml logs queen-rbee
    cleanup
    exit 1
fi

# Check rbee-hive
echo -n "  Checking rbee-hive... "
if curl -sf http://localhost:9200/v1/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅${NC}"
else
    echo -e "${RED}❌${NC}"
    echo -e "${RED}rbee-hive not healthy, check logs:${NC}"
    docker-compose -f docker-compose.integration.yml logs rbee-hive
    cleanup
    exit 1
fi

# Check mock-worker
echo -n "  Checking mock-worker... "
if curl -sf http://localhost:8001/v1/ready > /dev/null 2>&1; then
    echo -e "${GREEN}✅${NC}"
else
    echo -e "${RED}❌${NC}"
    echo -e "${RED}mock-worker not healthy, check logs:${NC}"
    docker-compose -f docker-compose.integration.yml logs mock-worker
    cleanup
    exit 1
fi

echo ""
echo -e "${GREEN}✅ All services healthy${NC}"
echo ""

# Run integration tests
echo -e "${BLUE}🧪 Running integration tests...${NC}"
echo ""

cd "$SCRIPT_DIR"

if [[ -n "$TAGS" ]]; then
    echo -e "${BLUE}Tags: $TAGS${NC}"
    cargo test --test cucumber -- --tags "$TAGS"
else
    cargo test --test cucumber -- --tags @integration
fi

TEST_STATUS=$?

echo ""

# Show service logs if tests failed
if [[ $TEST_STATUS -ne 0 ]]; then
    echo -e "${RED}❌ Tests failed, showing service logs:${NC}"
    echo ""
    echo -e "${YELLOW}=== Queen-rbee logs ===${NC}"
    docker-compose -f docker-compose.integration.yml logs --tail=50 queen-rbee
    echo ""
    echo -e "${YELLOW}=== Rbee-hive logs ===${NC}"
    docker-compose -f docker-compose.integration.yml logs --tail=50 rbee-hive
    echo ""
    echo -e "${YELLOW}=== Mock-worker logs ===${NC}"
    docker-compose -f docker-compose.integration.yml logs --tail=50 mock-worker
fi

# Cleanup
echo ""
echo -e "${YELLOW}🧹 Stopping services...${NC}"
cleanup

if [[ $TEST_STATUS -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    ✅ SUCCESS ✅                               ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                    ❌ FAILED ❌                                ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════════╝${NC}"
    exit 1
fi
