#!/usr/bin/env bash

: "${PNPM_HOME:="$PWD/.pnpm"}"
: "${COREPACK_HOME:="$PNPM_HOME"}"
: "${PLAYWRIGHT_BROWSERS_PATH:="$PWD/.cache/playwright"}"

export PNPM_HOME
export COREPACK_HOME
export PLAYWRIGHT_BROWSERS_PATH

mkdir -p "$PNPM_HOME/bin"
export PATH="$PNPM_HOME/bin:$PATH"

if [ -n "${PNPM_VERSION:-}" ]; then
  corepack install --global "pnpm@${PNPM_VERSION}" >/dev/null 2>&1 || true
else
  # Let corepack infer the pnpm version from package.json's packageManager field
  corepack install --global pnpm >/dev/null 2>&1 || true
fi

if [ ! -x "$PNPM_HOME/bin/pnpm" ]; then
  cat >"$PNPM_HOME/bin/pnpm" <<'EOF'
#!/usr/bin/env bash
exec corepack pnpm "$@"
EOF
  chmod +x "$PNPM_HOME/bin/pnpm"
fi

WRANGLER_CLI="wrangler"
if [ -n "${WRANGLER_VERSION:-}" ]; then
  WRANGLER_CLI="wrangler@${WRANGLER_VERSION}"
fi
cat >"$PNPM_HOME/bin/wrangler" <<EOF
#!/usr/bin/env bash
exec pnpm dlx ${WRANGLER_CLI} "\$@"
EOF
chmod +x "$PNPM_HOME/bin/wrangler"

TURBO_CLI="turbo"
if [ -n "${TURBO_VERSION:-}" ]; then
  TURBO_CLI="turbo@${TURBO_VERSION}"
fi
cat >"$PNPM_HOME/bin/turbo" <<EOF
#!/usr/bin/env bash
exec pnpm dlx ${TURBO_CLI} "\$@"
EOF
chmod +x "$PNPM_HOME/bin/turbo"
