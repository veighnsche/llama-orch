# Contradictions Found in 00_llama-orch.md

**Date**: 2025-10-03  
**Reviewed by**: Cascade AI  
**Status**: Requires resolution

---

## Critical Contradictions

### 1. ~~**Statelessness vs. Queue State**~~ ✅ **RESOLVED**
**Resolution**: Added [SYS-0022] Persistent State Store section with SQLite-based persistent state management.

The orchestrator now explicitly maintains:
- **Persistent state**: Job records, queue state, job history, tenant quotas (SQLite with WAL mode)
- **Ephemeral state**: Active SSE connections, cached pool states, in-flight requests

This resolves the contradiction by acknowledging the orchestrator is stateful and defining exactly what state is maintained, how it's persisted, and how restarts are handled.

---

### 2. **Tensor Parallelism vs. Worker Isolation** (MAJOR)
**Lines**: 258-270, 570-602

**Contradiction**: 
- Workers MUST be isolated processes (SYS-0041, SYS-0052)
- Workers MUST NOT share VRAM or CUDA contexts (line 271, 480)
- BUT: "tensor-parallel workers MUST coordinate" (line 570)
- AND: "shared GPU allocations across tensor-parallel shards" (line 572)

**Impact**: Tensor parallelism design is undefined and contradicts isolation requirements.

**Resolution Options**:
1. Single worker process using multiple GPUs (not 1:1 mapping)
2. Multiple coordinated workers (requires shared state/communication - violates isolation)
3. Defer tensor parallelism to M1 with separate design document

**Resolution**: Option 1 (single worker process using multiple GPUs) is chosen.

---

### 3. **Authentication in Home Mode** (MAJOR)
**Lines**: 734, 908-932

**Contradiction**:
- Home mode: "MUST accept requests from localhost without authentication" (line 908)
- Pool registration: "MUST be authenticated (see §8.1)" (line 734)

**Impact**: Unclear whether pool registration requires auth in home mode.

**Resolution Options**:
1. Pool registration is exempt from authentication in home mode (localhost)
2. Pool registration always requires authentication even in home mode
3. Authentication requirement only applies to platform mode

**Resolution**: Option 1 (exempt from authentication in home mode) is chosen.
Like if all the 3 binaries live in the same system then they don't need authentication, and they don't need audit. if the orchestrator lives on another machine. then authentication is enabled by default (make a policy that security goes before performance in platform mode and performance goes before security in home mode. maybe that'll eliminate some other contradictions). but in platform mode authentication is important to be on without being able to turn it off: platform mode is when a orchestrator connects to my platform so that an llama-orch sdk client can offload their tasks on your GPU's for a reward.

---

### 4. **Model Hot-Loading vs. Single-Model Workers** (MAJOR)
**Lines**: 261, 1270

**Contradiction**:
- Workers "MUST load exactly ONE model at startup" (SYS-0040, line 261)
- M1 feature: "Model hot-loading (swap models without restart)" (line 1270)

**Impact**: Feature roadmap contradicts core worker design.

**Resolution Options**:
1. Remove "without restart" - hot-loading means fast worker replacement
2. Clarify this is pool-level hot-loading (stop/start workers quickly)
3. Defer this feature and mark as "under design"

**Resolution**: Option 1 (Remove "without restart") is chosen.
Option 1 is the clear winner.
You don’t need in-process hot-swaps; you just need to clarify the language:

“Model hot-loading” = pool-level optimization: models are preloaded into host RAM or page cache, allowing workers for those models to start rapidly. Workers themselves remain single-model processes and are replaced when a model changes.

---

## Moderate Contradictions

### 5. **Quota Enforcement: Reject vs. Queue**
**Lines**: 979-988

**Contradiction**: "MUST reject or queue work when quotas are exceeded" - these are mutually exclusive actions.

**Resolution**: Clarify whether quota enforcement:
- Happens at admission (before queue) → reject with 429
- Happens at scheduling (after queue) → queue but don't dispatch
- Is configurable per tenant

**Resolution**: Never reject. make an infinite cue if needed (NO common sense limits please haha). we also need some sort of cron job to re-evaluate the current state of all the pools to see if there could be made any improvements. which should be a completely new feature.

---

### 6. **Retry and Determinism**
**Lines**: 887-899

**Contradiction**: "retries MUST NOT violate...determinism guarantees" but retrying on a different worker/GPU could produce different results.

**Resolution**: Clarify whether:
- Retries use the same worker/GPU (may not be available)
- Cross-worker retries may break determinism (acknowledge limitation)
- Determinism must be robust across workers (very hard to guarantee)

**Resolution**: You know what thing is. And I should mention it now. We are making the assumption that Models are deterministic. What if the core model that we put inference on. cannot be. deterministic. that would destroy SO many assumptions in the entire system. so Let's STOP assuming that models are deterministic. and let's do a research document for this. BUT if there is 1 model in the world that CAN be deterministic. then we can use that model to prove that our system is deterministic. Therefor WE KNOW that the system CANNOT be deterministic for now. BUT WE MAKE OUR SYSTEM AS IF IT IS DETERMINISTIC. 

---

### 7. **Tenant Metrics vs. Billing**
**Lines**: 1000-1012

**Contradiction**: 
- Metrics "SHOULD include tenant_id...MAY hash or omit" (line 1011)
- "Usage accounting MUST be recorded per tenant" (line 1012)

Hashed/omitted tenant_id prevents per-tenant aggregation.

**Resolution**: Separate concerns:
- Metrics (Prometheus) MAY omit/hash tenant_id to reduce cardinality
- Billing/accounting logs MUST include plaintext tenant_id

**Resolution**: First of all this is purely a platform thing right. for the home mode this is just useless overhead.
Because tenancy only in platform mode. We actually need to figure out the entire security model for the platform mode.

---

## Minor Issues / Clarifications Needed

### 8. **Eviction Policy Undefined**
**Lines**: 80, 181, 194, 447, 1141, 1159-1167

**Issue**: "eviction_policy" is mentioned but never defined. What is being evicted? When? How?

**Resolution**: Add section defining eviction scenarios, triggers, and semantics.

**Resolution**: Eviction could either be when a model is no longer needed in the RAM. OR when a worker in the VRAM needs to make room for another worker.

---

### 9. **Token Budgets Undefined**
**Lines**: 312, 707-714

**Issue**: "token budgets" mentioned but not defined. Is this max_tokens validation, per-tenant quotas, or context window validation?

**Resolution**: Define "token budgets" in glossary or earlier section.

**Resolution**: I have no idea. but I do assume that if a consumer client through the SDK using my platform is working of a credit system. and credit can run out. maybe the token budget should only be enabled when the user does not have auto-credit-refill on? somethign like that. but also when the user puts a token limit? something like that. We really need to make a real distinction between home mode (everything in one system) - lab mode (orchestrator and poolmanagerd in different systems) - platform mode (Security first mode that allowes multiple strangers to safely and privately run inference on your GPU's through my platform. for a reward.)

---

### 10. **Backoff Parameters Unspecified**
**Lines**: 432-440

**Issue**: "retries MUST apply backoff" but no parameters specified (initial delay, multiplier, ceiling, max attempts).

**Resolution**: Add backoff configuration section or reference.

**Resolution**: I have no idea what would be best. I also mentioned that we should make an infinite queue. so that contradicts backoff logic. I think backoff is better than infinite queue logic. you specify the parameters

---

### 11. **Temperature Parameter**
**Lines**: 307, 698-704

**Issue**: Example shows `temperature: 0.7` but requirements don't list it. Is it optional? Default value?

**Resolution**: Clarify if temperature is optional with default, or required for non-deterministic sampling.

**Resolution**: Like the SDK need to have an OPENAI API facade as default. so that users can always use llama-orch as a drop-in for exising client systems. Maybe you need to take a look in /home/vince/Projects/llama-orch/contracts/openapi for a better reference. also that openaAPI folder is old. that one needs updating. but it does hopefully explain enough. /home/vince/Projects/llama-orch/consumers/llama-orch-sdk here is the SDK. if it does not mention in the specs that it has an openai API facade in it. then it must me mentioned in there.

But the orchestrator API shows all the information that the user wants for a web-based UI to orchestrate manually. THat needs to be possible too. like we must use the logging as a sort of stream to the web-ui (also make a new WEB_UI thing in the frontend folder made with vue. a scaffold would be nice) so the web-ui connects with the sdk. who connects with orchestratord api. meaning that the orchestratord must emit all the logging so that the web-ui can interpret that as movement of facts in the system . does that make sense? also the user can also hold certain models in a certain GPU (like forbidden to evict or something) all that sorts of UX features so that the user has a web ui for orchestrating themselves. the web-ui is also a policy shaper.

---

### 12. **Platform Orchestrator Intelligence**
**Lines**: 622-626

**Issue**: "smart router" vs "Provider orchestrators make their own placement decisions" - what intelligence does platform orchestrator have?

**Resolution**: Clarify what routing decisions platform makes (provider selection) vs. what it delegates (worker selection).

**Resolution**: There is no more such thing as a router. the scheduler does the placement. the schedular does a lot the schedular is the only monolithic code pattern in the entire repo I believe. this is some important detail i want to see back. becasue the schedular is a policy machine. which decided where when what how etc all based on policy. so it's gonna have a lot of complexity centralized in the schedular. all tightly coupled. 

We used to try to keep them loosly coupled. like routing and placement etc. but it's all scheduling. and they all work together tightly

---

### 13. **Determinism and Model Pinning**
**Lines**: 648-658

**Issue**: "SHOULD pin @rev" and "SHOULD pin artifact" but determinism (SYS-0003) is a MUST. Inconsistent RFC-2119 levels.

**Resolution**: Either strengthen to MUST or clarify determinism only applies when pinned.

**Resolution**: I have already mentioned what I think about determinism. please adhere to my previous comment 

---

### 14. **Worker Startup Latency**
**Lines**: 840-848

**Issue**: "Worker startup SHOULD complete within 60s" but this includes model loading which varies by model size. Not directly controllable.

**Resolution**: Consider whether this should be a SHOULD or just a target/guideline.

**Resolution**: This is something that the orchestratord MUST decide with their performance logic. It might even be so complicated that It might even become some kind of browser based IDE programming langauge so that the user can define the policy programmatically. and we offer a object that contains all the facts about the system and the orchestratord just runs the scheduler code to do stuff. I think that's an interesting idea to dive into.

---

### 15. **Requeue and Statelessness**
**Lines**: 433-436

**Issue**: "requeue" implies orchestrator knows which job was on failed worker, but if stateless, how?

**Resolution**: Acknowledge job-to-worker assignment state is maintained.

**Resolution**: not stateless anymore. and there is now a cron-job that checks if things can be neater if there is a big queue. so the orchestratord must raise a flag to start or stop the optimizer cron job. see the orchestratord centralizes all the complexity. therefor it might just be a better idea to give the user the tools to program the schedular for the home and lab mode. (not platform mode, no we need control in platform mode.)

---

### 16. **SSE Resume and Statelessness**
**Lines**: 702-720

**Issue**: "resume streaming from last sent offset" requires tracking state per job_id, contradicts stateless claim.

**Resolution**: Acknowledge SSE state is maintained for active streams.

**Resolution**: We have a state now

---

### 17. **No Direct Worker Communication**
**Lines**: 129-134

**Issue**: Heading "No Direct Worker Communication" but orchestrator directly calls workers. Misleading.

**Resolution**: Clarify heading means "No Direct CLIENT→Worker Communication".

**Resolution**: THing is. when the poolmanager spawns a worker. the poolworker will receive the http endpoin from the worker. then when the worker signals to the poolworker that it is ready. then the poolmanager exposes the workers http endpoint to the orchestrator. then the orchestrator can call the worker directly.

---

## Summary Statistics

- **Critical contradictions**: 3 (1 resolved)
- **Moderate contradictions**: 3
- **Minor issues/clarifications**: 10
- **Total issues found**: 17
- **Resolved**: 1

## Recommendations

1. **Immediate**: Resolve critical contradictions (1-4) before implementation
2. **High Priority**: Address moderate contradictions (5-7) in next spec revision
3. **Medium Priority**: Clarify minor issues (8-17) to improve spec quality
4. **Process**: Consider adding a glossary section for undefined terms

---

**Next Steps**:
1. Review each contradiction with team
2. Make architectural decisions where design is unclear
3. Update spec with resolutions
4. Remove contradiction comments once resolved
5. Add glossary/definitions section for clarity
