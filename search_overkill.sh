#!/usr/bin/env bash
# Home Profile reduction helper — locate/deprecate enterprise features before manual edits.
# NOTE: This script is non-destructive by default. It uses grep to surface matches.
#       Suggested sed -i lines are provided commented-out for you to apply intentionally.

set -euo pipefail

root_dir=$(cd "$(dirname "$0")" && pwd)
cd "$root_dir"

# Recursive grep helper excluding build artifacts (target at any depth)
rgx() {
  grep -R -n --color=never \
    --exclude-dir=target \
    --exclude-dir='*/target' \
    --exclude-dir='**/target' \
    "$@"
}

# --- Multi-tenancy / RBAC / quotas ------------------------------------------------------------
# Locate references to multi-tenancy, tenants, RBAC, and quotas across docs/spec/code.
echo "[scan] multi-tenancy / RBAC / quotas"
rgx -E "multi-tenant|multi tenancy|tenancy|tenant|RBAC|quota" -- . || true

# Example (commented): remove tenant admission share metric from metrics.lint.json
# sed -i '/"admission_share"/,+5d' ci/metrics.lint.json

# --- Fairness (WFQ), deadlines (EDF), preemption, resumable jobs -------------------------------
echo "[scan] fairness / deadlines / preemption / resumable jobs"
rgx -E "fairness|WFQ|deadline|EDF|preempt|resum(e|ption)|resumptions_total|preemptions_total|DEADLINE_UNMET" -- . || true

# Example (commented): drop fairness/preemption config structs from config schema
# sed -i '/struct FairnessConfig/,/}/d' contracts/config-schema/src/lib.rs
# sed -i '/struct PreemptionConfig/,/}/d' contracts/config-schema/src/lib.rs

# --- Lifecycle states (reduce to Active | Retired) ---------------------------------------------
echo "[scan] lifecycle states beyond Active/Retired"
rgx -E '\\bDraft\\b|\\bDeprecated\\b|\\bCanary\\b|percent rollout|rollout' -- . || true

# Example (commented): narrow ModelState to Active|Retired
# sed -i 's/Draft/Active/g' orchestratord/src/state.rs
# sed -i '/Deprecated/d' orchestratord/src/state.rs

# --- Trust / SBOM / signatures -----------------------------------------------------------------
echo "[scan] trust & SBOM/signatures enforcement"
rgx -E "SBOM|signature|signatures|trust[_ -]?policy|require_signature|require_sbom|artifact registry|artifacts" -- . || true

# Example (commented): make trust policy optional-only in config schema
# sed -i '/TrustPolicyConfig/,/}/d' contracts/config-schema/src/lib.rs

# --- Advanced metrics to drop ------------------------------------------------------------------
echo "[scan] advanced metrics (admission_share, deadlines_met_ratio, preemptions/resumptions, latency histograms)"
rgx -E "admission_share|deadlines_met_ratio|preemptions_total|resumptions_total|latency_first_token_ms|latency_decode_ms" -- . || true

# Example (commented): remove metrics from registry and linter
# sed -i '/ADMISSION_SHARE/,+3d' orchestratord/src/metrics.rs
# sed -i '/DEADLINES_MET_RATIO/,+3d' orchestratord/src/metrics.rs
# sed -i '/preemptions_total/,+5d' ci/metrics.lint.json
# sed -i '/resumptions_total/,+5d' ci/metrics.lint.json
# sed -i '/latency_first_token_ms/,+5d' ci/metrics.lint.json
# sed -i '/latency_decode_ms/,+5d' ci/metrics.lint.json

# --- APIs: correlation IDs, policy_label in 429, artifacts, drains, CDC ------------------------
echo "[scan] correlation IDs header (X-Correlation-Id)"
rgx "X-Correlation-Id" -- . || true

echo "[scan] policy_label advisory fields in 429 bodies"
rgx "policy_label" -- . || true

echo "[scan] artifact registry endpoints in OpenAPI/contracts"
rgx -E "/v1/artifacts|artifact(s)?" -- contracts/openapi || true

echo "[scan] drain endpoints in control plane"
rgx -E "/v1/pools/.*/drain|drain_pool" -- contracts/openapi orchestratord || true

# Example (commented): remove correlation ID injection in handlers and provider tests
# sed -i '/X-Correlation-Id/d' orchestratord/src/http/data.rs
# sed -i '/X-Correlation-Id/d' orchestratord/src/http/control.rs
# sed -i '/X-Correlation-Id/d' orchestratord/tests/provider_verify.rs

# --- Ops: canaries, HA, percent rollouts -------------------------------------------------------
echo "[scan] canaries / HA / percent rollouts"
rgx -E "canary|\\bHA\\b|high availability|percent rollout" -- . || true

# --- Placement: advanced hints -----------------------------------------------------------------
echo "[scan] placement hints (NUMA/PCIe/topology/tensor_split)"
rgx -E "\\bNUMA\\b|PCIe|\\bPCI\\b|tensor_split|topology" -- . || true

# --- Security: bind address (ensure loopback) --------------------------------------------------
echo "[scan] non-loopback binds (consider forcing 127.0.0.1)"
rgx -E '\\b0\\.0\\.0\\.0\\b|::' -- orchestratord || true

# --- OpenAPI removals to perform (manual) ------------------------------------------------------
echo "[hint] Edit contracts/openapi/control.yaml to remove /v1/artifacts* and /v1/pools/{id}/drain"
echo "[hint] Edit contracts/openapi/data.yaml to remove policy_label and correlation ID mentions in examples"

echo "[done] Scans complete. Review output and apply commented sed -i lines selectively."
