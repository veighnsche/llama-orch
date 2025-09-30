# README Update Progress

**Started**: 2025-10-01  
**Strategy**: Update in priority order, one at a time  
**Goal**: Make each README self-contained (no references to deleted .md files)

---

## Progress: 5/67 Complete

### ✅ Done (5)
- [x] `./README.md` (root)
- [x] `./bin/orchestratord/README.md` ✨
- [x] `./bin/pool-managerd/README.md` ✨
- [x] `./bin/orchestratord/bdd/README.md` ✨
- [x] `./bin/pool-managerd/bdd/README.md` ✨

### 🚧 In Progress (0)

### ⏳ Pending (62)

---

## ✅ P0: Core Binaries (COMPLETE!)

- [x] `./bin/orchestratord/README.md` ✅
- [x] `./bin/pool-managerd/README.md` ✅
- [x] `./bin/orchestratord/bdd/README.md` ✅
- [x] `./bin/pool-managerd/bdd/README.md` ✅

## P1: Core Libraries (6 files)

- [ ] `./libs/orchestrator-core/README.md`
- [ ] `./libs/catalog-core/README.md`
- [ ] `./libs/adapter-host/README.md`
- [ ] `./libs/proof-bundle/README.md`
- [ ] `./libs/orchestrator-core/bdd/README.md`
- [ ] `./libs/catalog-core/bdd/README.md`

## P2: Multi-Node Libraries (4 files)

- [ ] `./libs/control-plane/service-registry/README.md`
- [ ] `./libs/gpu-node/handoff-watcher/README.md`
- [ ] `./libs/gpu-node/node-registration/README.md`
- [ ] `./libs/shared/pool-registry-types/README.md`

## P3: Worker Adapters (10 files)

- [ ] `./libs/worker-adapters/README.md`
- [ ] `./libs/worker-adapters/adapter-api/README.md`
- [ ] `./libs/worker-adapters/http-util/README.md`
- [ ] `./libs/worker-adapters/llamacpp-http/README.md`
- [ ] `./libs/worker-adapters/vllm-http/README.md`
- [ ] `./libs/worker-adapters/tgi-http/README.md`
- [ ] `./libs/worker-adapters/triton/README.md`
- [ ] `./libs/worker-adapters/openai-http/README.md`
- [ ] `./libs/worker-adapters/mock/README.md`
- [ ] `./libs/worker-adapters/http-util/.proof_bundle/README.md`

## P4: Supporting Libraries (9 files)

- [ ] `./libs/provisioners/engine-provisioner/README.md`
- [ ] `./libs/provisioners/model-provisioner/README.md`
- [ ] `./libs/observability/narration-core/README.md`
- [ ] `./libs/auth-min/README.md`
- [ ] `./contracts/api-types/README.md`
- [ ] `./contracts/config-schema/README.md`
- [ ] `./libs/provisioners/engine-provisioner/bdd/README.md`
- [ ] `./libs/provisioners/model-provisioner/bdd/README.md`
- [ ] `./libs/observability/narration-core/bdd/README.md`

## P5: Test Harness (5 files)

- [ ] `./test-harness/bdd/README.md`
- [ ] `./test-harness/chaos/README.md`
- [ ] `./test-harness/determinism-suite/README.md`
- [ ] `./test-harness/e2e-haiku/README.md`
- [ ] `./test-harness/metrics-contract/README.md`

## P6: Tools & Consumers (16 files)

- [ ] `./tools/openapi-client/README.md`
- [ ] `./tools/spec-extract/README.md`
- [ ] `./tools/readme-index/README.md`
- [ ] `./xtask/README.md`
- [ ] `./consumers/llama-orch-sdk/README.md`
- [ ] `./consumers/llama-orch-utils/README.md`
- [ ] `./consumers/llama-orch-utils/src/fs/file_reader/README.md`
- [ ] `./consumers/llama-orch-utils/src/fs/file_writer/README.md`
- [ ] `./consumers/llama-orch-utils/src/llm/invoke/README.md`
- [ ] `./consumers/llama-orch-utils/src/model/define/README.md`
- [ ] `./consumers/llama-orch-utils/src/orch/response_extractor/README.md`
- [ ] `./consumers/llama-orch-utils/src/params/define/README.md`
- [ ] `./consumers/llama-orch-utils/src/prompt/message/README.md`
- [ ] `./consumers/llama-orch-utils/src/prompt/thread/README.md`
- [ ] `./consumers/.examples/M002-pnpm/README.md`

## P7: Frontend (3 files)

- [ ] `./frontend/bin/commercial-frontend/README.md`
- [ ] `./frontend/libs/storybook/README.md`
- [ ] `./frontend/bin/commercial-frontend/.docs/sections/README.md`

## P8: Templates (10 files) **SKIP THESE ARE GENERATED FILES**

- [ ] `./.proof_bundle/README.md`
- [ ] `./.proof_bundle/templates/README.md`
- [ ] `./.proof_bundle/templates/bdd/README.md`
- [ ] `./.proof_bundle/templates/chaos/README.md`
- [ ] `./.proof_bundle/templates/contract/README.md`
- [ ] `./.proof_bundle/templates/determinism/README.md`
- [ ] `./.proof_bundle/templates/e2e-haiku/README.md`
- [ ] `./.proof_bundle/templates/home-profile-smoke/README.md` ⚠️ RENAME
- [ ] `./.proof_bundle/templates/integration/README.md`
- [ ] `./.proof_bundle/templates/unit/README.md`

---

## Update Checklist (Per README)

- [ ] Remove references to .md files (except README.md)
- [ ] Remove HOME_PROFILE / CLOUD_PROFILE terminology
- [ ] Add clear "What This Does" section
- [ ] Add usage examples (code or commands)
- [ ] Add test commands
- [ ] Make self-contained (no external doc dependencies)
- [ ] Keep spec references (they're in .specs/ which stays)
- [ ] Update architecture diagrams if needed
