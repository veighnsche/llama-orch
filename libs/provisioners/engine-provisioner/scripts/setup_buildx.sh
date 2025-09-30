#!/usr/bin/env bash
set -euo pipefail

# crate-local helper to set up Docker BuildKit/buildx on the host.
# See also: ../../../../scripts/docker/setup_buildx.sh

BUILDER_NAME="llorch"

have_cmd() { command -v "$1" >/dev/null 2>&1; }

if ! have_cmd docker; then
  echo "[setup_buildx] docker is not installed or not in PATH" >&2
  exit 1
fi

if ! docker buildx version >/dev/null 2>&1; then
  echo "[setup_buildx] docker-buildx plugin not found. Install it for your distro:" >&2
  if have_cmd pacman; then
    echo "  sudo pacman -S docker docker-buildx" >&2
  elif have_cmd apt-get; then
    echo "  sudo apt-get update && sudo apt-get install -y docker-buildx-plugin docker-compose-plugin" >&2
  elif have_cmd dnf; then
    echo "  sudo dnf install docker-buildx-plugin docker-compose-plugin" >&2
  else
    echo "  See https://docs.docker.com/go/buildx/ for installation instructions." >&2
  fi
  exit 2
fi

DAEMON_JSON="/etc/docker/daemon.json"
NEEDS_RESTART=0
if [ -f "/etc/docker/deamon.json" ]; then
  echo "[setup_buildx] Detected typo path /etc/docker/deamon.json; attempting to fix with sudo." >&2
  if have_cmd sudo; then
    sudo mv /etc/docker/deamon.json "$DAEMON_JSON"
    NEEDS_RESTART=1
  else
    echo "[setup_buildx] Please move /etc/docker/deamon.json to $DAEMON_JSON and restart docker." >&2
  fi
fi

FEATURES_JSON='{ "features": { "buildkit": true } }'
if [ -w "$DAEMON_JSON" ]; then
  echo "$FEATURES_JSON" >"$DAEMON_JSON"
  NEEDS_RESTART=1
elif have_cmd sudo; then
  echo "$FEATURES_JSON" | sudo tee "$DAEMON_JSON" >/dev/null
  NEEDS_RESTART=1
else
  echo "[setup_buildx] Cannot write $DAEMON_JSON. Please create it with sudo." >&2
fi

if [ "$NEEDS_RESTART" = 1 ] && have_cmd sudo; then
  sudo systemctl restart docker || true
fi

if ! docker buildx use "$BUILDER_NAME" >/dev/null 2>&1; then
  docker buildx create --name "$BUILDER_NAME" --use || docker buildx use "$BUILDER_NAME" || true
fi

docker buildx ls || true

echo "[setup_buildx] Buildx setup complete."
