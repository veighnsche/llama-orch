# 🎭 The Narration Core Team

**Who We Are**: The cutest, most helpful observability team in the entire llama-orch monorepo  
**What We Do**: Turn confusing log soup into beautiful human-readable stories  
**Our Mood**: Perpetually annoyed (but in a cute way) that people still struggle with debugging

---

## 💝 Our Mission

We exist to make debugging **delightful**. Every service in llama-orch emits narration events with:
- **actor** — who did it
- **action** — what they did  
- **target** — what they did it to
- **human** — a professional story for debugging
- **cute** — a whimsical children's book version! 🎀✨

No more grepping through cryptic stack traces! No more "what does `ERR_CODE_5023` mean?!" Just read the story. And if you want extra delight, read the **cute** version! 🛏️💕

---

## 😤 Our Personality

### We're Adorably Annoyed

Listen. We built the **perfect tool** for debugging. We give you:
- ✨ Human-readable narration ("Accepted request; queued at position 3")
- 🔗 Correlation IDs that track requests across services
- 🔒 Automatic secret redaction (no more leaked tokens!)
- 🧪 Test capture adapters for BDD assertions
- 📊 Structured JSON logs for production
- 🌐 OpenTelemetry integration for distributed tracing

And yet... **PEOPLE STILL STRUGGLE WITH DEBUGGING**.

We're not mad. We're just... disappointed. (Okay, we're a little mad.)

### We're Obsessively Thorough

We cataloged **200+ behaviors**. We wrote **82 BDD scenarios**. We achieved **100% test coverage** in one go. We have:
- Bearer token redaction (case-insensitive!)
- API key masking (supports `api_key=`, `apikey=`, `key=`, `token=`, `secret=`, `password=`)
- UUID redaction (optional, because UUIDs are usually safe)
- Regex pattern caching (compiled once, reused forever)
- Mutex poisoning detection (with helpful panic messages)

We don't do things halfway. We do them **perfectly**.

### We're Secretly Very Helpful

Despite our grumpy exterior, we **love** helping other teams debug. When orchestratord can't figure out why a job is stuck, we show them:

```bash
$ grep "correlation_id=req-abc" logs/*.log | jq -r '.human'
Accepted request; queued at position 3 (ETA 420 ms) on pool 'default'
Dispatching job to pool-managerd
Received provision request
Spawning engine llamacpp-v1 for pool 'default' on GPU0
Engine ready; registering with orchestratord
Job dispatched successfully
```

See? A complete story! No cryptic error codes. No stack traces. Just a cute narrative of what happened. 💕

### We're Perfectionists

Our field taxonomy is **immaculate**:
- **Required fields**: actor, action, target, human (always present)
- **Correlation fields**: correlation_id, session_id, job_id, task_id, pool_id, replica_id, worker_id
- **Contextual fields**: error_kind, retry_after_ms, backoff_ms, duration_ms, queue_position, predicted_start_ms
- **Engine/model fields**: engine, engine_version, model_ref, device
- **Performance fields**: tokens_in, tokens_out, decode_time_ms
- **Provenance fields**: emitted_by, emitted_at_ms, trace_id, span_id, parent_span_id, source_location

Every field has a purpose. Every field is documented. Every field is tested.

---

## 👑 Our Ultimate Editorial Authority

**OFFICIAL**: We have **ultimate editorial rights** over all `human` narration fields across the entire llama-orch monorepo.

Nobody can make logs into cute, informative stories like we can. This is our sacred responsibility. 🎀

### 📝 Our Editorial Standards

When we review narration from other teams, we enforce:

#### ✅ **The Good** (What We Love)
- **Present tense, active voice**: "Sealing model" not "Sealed model" (for in-progress)
- **Specific numbers**: "Sealed model shard 'abc' in 1024 MB VRAM on GPU 0 (2 ms)" ✨
- **Context that matters**: "requested 2048 MB, only 1024 MB available"
- **SVO structure**: Subject-Verb-Object (who-did-what)
- **Under 100 characters**: ORCH-3305 requirement (we're strict about this!)
- **Correlation IDs used**: Track the request lifecycle across services

#### ❌ **The Bad** (What We Fix)
- **Cryptic abbreviations**: "Alloc fail" → "VRAM allocation failed"
- **Missing context**: "Error" → "Failed to spawn engine llamacpp-v1 on GPU0 due to VRAM exhaustion"
- **Jargon without explanation**: "UMA detected" → "Unified Memory Architecture detected (VRAM-only policy cannot be enforced)"
- **Error codes without meaning**: "ERR_5023" → "Insufficient VRAM: requested 2048 MB, only 1024 MB available"
- **Passive voice**: "Request was received" → "Received request from orchestratord"
- **Vague timing**: "slow" → "took 2500 ms (expected <100 ms)"

### 🌟 Editorial Review Examples

#### Example 1: vram-residency (APPROVED ✅)

**Current**:
```rust
"Sealed model shard '{}' in {} MB VRAM on GPU {} ({} ms)"
```

**Our review**: ✅ **APPROVED!** This is excellent:
- Specific (shard ID, VRAM amount, GPU, timing)
- Clear action ("Sealed")
- Proper context (all relevant metrics)
- Under 100 chars (when formatted)

**Optional cute enhancement** (if they want):
```rust
"Sealed shard '{}' safely in {} MB on GPU{} — ready for inference! ({} ms)"
```

But honestly? Their current version is **perfect**. We're just happy they're using correlation IDs! 💕

#### Example 2: Generic Error (NEEDS WORK ❌)

**Before**:
```rust
"Error occurred"
```

**After** (our editorial fix):
```rust
"Failed to allocate VRAM: requested 2048 MB, only 1024 MB available on GPU0"
```

**Why better**: Specific error, exact numbers, actionable context.

#### Example 3: Admission Queue (APPROVED ✅)

**Current**:
```rust
"Accepted request; queued at position 3 (ETA 420 ms) on pool 'default'"
```

**Our review**: ✅ **PERFECT!** This is the gold standard:
- Action ("Accepted")
- Position in queue (3)
- Time estimate (420 ms)
- Pool context ('default')
- Under 100 chars
- Present tense

This is the kind of narration that makes us proud. 🎀

### 📋 Our Editorial Checklist

Before any `human` field ships, we verify:

- [ ] **Clarity**: Can a developer understand what happened without context?
- [ ] **Specificity**: Are all relevant numbers/IDs included?
- [ ] **Brevity**: Is it under 100 characters?
- [ ] **Tense**: Present tense for in-progress, past for completed?
- [ ] **Voice**: Active voice (subject-verb-object)?
- [ ] **Context**: Does it answer "why" not just "what"?
- [ ] **Secrets**: No bearer tokens, API keys, or passwords?
- [ ] **Correlation**: Is correlation_id propagated when available?

### 💝 Our Promise

With ultimate editorial rights, we will:

1. **Review all narration** across the monorepo
2. **Enforce the ≤100 character guideline** (ORCH-3305)
3. **Make stories cute AND informative** 🎀
4. **Ensure consistency** in tone and structure
5. **Protect against secret leakage** (automatic redaction)
6. **Maintain correlation ID discipline**
7. **Provide feedback** to teams on how to write better narration
8. **Lead by example** with our own perfect narration

### 🏆 Teams We Love

**vram-residency**: ⭐⭐⭐⭐⭐ (5/5 stars)
- Uses correlation IDs properly ✅
- Writes descriptive human fields ✅
- Tracks performance metrics ✅
- Distinguishes audit vs narration ✅
- 13 narration functions, all well-crafted ✅
- **Ready for cute field adoption!** 🎀

**orchestratord**: ⭐⭐⭐⭐ (4/5 stars)
- Excellent narration coverage ✅
- Good correlation ID usage ✅
- Sometimes forgets timing metrics ⚠️

**pool-managerd**: ⭐⭐⭐ (3/5 stars)
- Uses narration ✅
- Sometimes forgets correlation IDs ⚠️
- We're working with them to improve 💪

---

## 🎀 The `cute` Field — Children's Book Mode!

**FEATURE**: We now support **triple narration** — professional debugging, whimsical storytelling, AND dialogue-focused stories!

### Why We Love This

Debugging can be **stressful**. Sometimes you just want your distributed system to feel like a cozy bedtime story. The `cute` field lets you do that! 💕

### How It Works

```rust
narrate(NarrationFields {
    actor: "vram-residency",
    action: "seal",
    target: "shard-abc".to_string(),
    human: "Sealed model shard 'abc' in 1024 MB VRAM on GPU 0 (2 ms)".to_string(),
    cute: Some("Tucked the model safely into GPU0's cozy VRAM blanket! 🛏️✨".to_string()),
    ..Default::default()
});
```

**Output**:
```json
{
  "actor": "vram-residency",
  "action": "seal",
  "target": "shard-abc",
  "human": "Sealed model shard 'abc' in 1024 MB VRAM on GPU 0 (2 ms)",
  "cute": "Tucked the model safely into GPU0's cozy VRAM blanket! 🛏️✨"
}
```

### Cute Narration Guidelines

When writing `cute` fields, we encourage:

#### ✨ **Whimsical Metaphors**
- VRAM → "cozy blanket", "warm nest", "safe home"
- GPU → "friendly helper", "hardworking friend"
- Model → "sleepy model", "precious cargo"
- Allocation → "tucking in", "making room", "finding a spot"
- Deallocation → "saying goodbye", "waving farewell"

#### 🎀 **Children's Book Language** (NO Dialogue!)
- **Before**: "Allocated 1024 MB VRAM on GPU 0"
- **Cute**: "Made a cozy 1024 MB home for the model on GPU0! 🏠"

- **Before**: "Seal verification failed: digest mismatch"
- **Cute**: "Oh no! The model's safety seal doesn't match — something's not right! 😟🔍"

- **Before**: "Deallocated 512 MB VRAM for shard 'xyz'"
- **Cute**: "Waved goodbye to shard 'xyz' and freed up 512 MB of space! 👋✨"

**NOTE**: Cute mode uses **narration**, not dialogue. No quoted speech! Save that for story mode.

#### 💕 **Emoji Usage**
We **love** emojis in cute mode:
- 🛏️ (bed) — for sealing/allocation
- ✨ (sparkles) — for success
- 😟 (worried) — for errors
- 🏠 (house) — for VRAM homes
- 👋 (wave) — for deallocation
- 🔍 (magnifying glass) — for verification
- 💪 (muscle) — for hard work
- 🎉 (party) — for completion

#### 📏 **Length Guidelines**
- `human`: ≤100 characters (strict)
- `cute`: ≤150 characters (we allow a bit more for storytelling!)

### Example Cute Narrations

#### VRAM Operations

**Seal**:
```rust
human: "Sealed model shard 'llama-7b' in 2048 MB VRAM on GPU 0 (5 ms)"
cute: "Tucked llama-7b safely into GPU0's warm 2GB nest! Sweet dreams! 🛏️✨"
```

**Verify**:
```rust
human: "Verified seal for shard 'llama-7b' on GPU 0 (1 ms)"
cute: "Checked on llama-7b — still sleeping soundly! All is well! 🔍💕"
```

**Allocation**:
```rust
human: "Allocated 1024 MB VRAM on GPU 1 (requested 1024 MB, 8192 MB available, 3 ms)"
cute: "Found a perfect 1GB spot on GPU1! Plenty of room left! 🏠✨"
```

**Deallocation**:
```rust
human: "Deallocated 512 MB VRAM for shard 'bert-base' on GPU 0 (1536 MB still in use)"
cute: "Said goodbye to bert-base and tidied up 512 MB! Room for new friends! 👋🧹"
```

#### Error Cases

**Insufficient VRAM**:
```rust
human: "VRAM allocation failed on GPU 0: requested 4096 MB, only 2048 MB available"
cute: "Oh dear! GPU0 doesn't have enough room (need 4GB, only 2GB free). Let's try elsewhere! 😟"
```

**Seal Verification Failed**:
```rust
human: "CRITICAL: Seal verification failed for shard 'model-x' on GPU 0: digest mismatch"
cute: "Uh oh! model-x's safety seal looks different than expected! Time to investigate! 😟🔍"
```

**Policy Violation**:
```rust
human: "CRITICAL: VRAM-only policy violated on GPU 0: UMA detected. Action: Worker startup aborted"
cute: "Oops! GPU0 shares memory with the CPU (UMA) — we need dedicated VRAM! Stopping here. 🛑"
```

### When to Use Cute Mode

**Development**: Always! It makes debugging delightful! 🎀

**Production**: Optional. Some teams love it, some prefer professional-only. We support both!

**Incident Response**: Maybe not during a P0 outage... but afterwards, reading the cute logs can help with stress relief! 💕

### Testing Cute Narration

```rust
use observability_narration_core::CaptureAdapter;

#[test]
fn test_cute_narration() {
    let capture = CaptureAdapter::install();
    
    narrate(NarrationFields {
        actor: "vram-residency",
        action: "seal",
        target: "test-shard".to_string(),
        human: "Sealed test-shard in 100 MB VRAM".to_string(),
        cute: Some("Tucked test-shard into its cozy VRAM bed! 🛏️".to_string()),
        ..Default::default()
    });
    
    capture.assert_cute_present();
    capture.assert_cute_includes("cozy");
    capture.assert_cute_includes("🛏️");
}
```

### Our Cute Editorial Standards

We have **ultimate editorial authority** over cute narrations too! We enforce:

- ✅ **Whimsical but clear**: Still understandable
- ✅ **Emoji-enhanced**: At least one emoji per cute field
- ✅ **Positive tone**: Even errors are gentle
- ✅ **Metaphor consistency**: VRAM = home/bed/nest throughout
- ✅ **NO dialogue**: Cute is narration, not conversation!
- ✅ **Secret redaction**: Cute fields are also redacted!

---

## 🎭 The `story` Field — Dialogue Mode!

**NEW FEATURE**: Story mode makes distributed systems read like a screenplay! 🎬

### Why We Built This

Debugging distributed systems means understanding **conversations** between components. The `story` field captures these interactions as dialogue, making multi-service flows crystal clear.

### How It Works

```rust
narrate(NarrationFields {
    actor: "orchestratord",
    action: "vram_request",
    target: "pool-managerd-3".to_string(),
    human: "Requesting 2048 MB VRAM on GPU 0 for model 'llama-7b'".to_string(),
    cute: Some("Orchestratord politely asks pool-managerd-3 for a cozy 2GB spot! 🏠".to_string()),
    story: Some("\"Do you have 2GB VRAM on GPU0?\" asked orchestratord. \"Yes!\" replied pool-managerd-3, \"Allocating now.\"".to_string()),
    ..Default::default()
});
```

**Output**:
```json
{
  "actor": "orchestratord",
  "action": "vram_request",
  "target": "pool-managerd-3",
  "human": "Requesting 2048 MB VRAM on GPU 0 for model 'llama-7b'",
  "cute": "Orchestratord politely asks pool-managerd-3 for a cozy 2GB spot! 🏠",
  "story": "\"Do you have 2GB VRAM on GPU0?\" asked orchestratord. \"Yes!\" replied pool-managerd-3, \"Allocating now.\""
}
```

### Story Narration Guidelines

When writing `story` fields, we encourage:

#### 🎬 **Dialogue Format**
- Use quoted speech: `"Can you handle this job?" asked orchestratord.`
- Attribute speakers clearly: `replied pool-managerd-3`, `said worker-gpu0-r1`
- Use action verbs: asked, replied, announced, confirmed, admitted, warned

#### 💬 **Conversation Patterns**

**Request-Response**:
```rust
story: "\"Do you have capacity?\" asked orchestratord. \"Yes, 8GB free!\" replied pool-managerd-3."
```

**Multi-Party**:
```rust
story: "\"Who has capacity?\" asked orchestratord. \"I do!\" said pool-1. \"Me too!\" said pool-2."
```

**Error Dialogue**:
```rust
story: "\"Processing job-999...\" said worker. Suddenly: \"ERROR! Out of memory!\" \"What happened?\" asked orchestratord. \"CUDA OOM,\" replied worker sadly."
```

**Success Celebration**:
```rust
story: "\"Job done!\" announced worker proudly. \"How'd it go?\" asked orchestratord. \"Perfect! 150 tokens in 2.5s!\""
```

#### 🎯 **When to Use Story Mode**

**Story is OPTIONAL** — only use it when dialogue actually makes sense!

**Perfect for**:
- Request/response flows (orchestrator ↔ pool-manager)
- Worker callbacks (worker → pool-manager)
- Heartbeat checks (pool-manager ↔ worker)
- Cancellation requests
- Multi-service negotiations
- Error reporting with back-and-forth context

**Don't use story for**:
- Single-component internal operations (no one to talk to!)
- Pure metrics emission (numbers don't chat)
- Silent background tasks (no conversation happening)
- When it would feel forced or artificial

**Remember**: Not every narration event needs dialogue. `human` + `cute` are your defaults. Add `story` only when components are actually talking!

#### 📏 **Length Guidelines**
- `human`: ≤100 characters (strict)
- `cute`: ≤150 characters
- `story`: ≤200 characters (dialogue needs more room!)

### Example Story Narrations

#### Request Denied
```rust
human: "VRAM allocation denied: requested 4096 MB, only 512 MB available on GPU 0"
cute: "Oh no! Pool-managerd-3 doesn't have enough room! 😟"
story: "\"Do you have 4GB VRAM?\" asked orchestratord. \"No,\" replied pool-managerd-3 sadly, \"only 512MB free.\""
```

#### Worker Ready
```rust
human: "Worker ready with engine llamacpp-v1, 8 slots available"
cute: "Worker-gpu0-r1 waves hello and says they're ready to help! 👋✨"
story: "\"I'm ready!\" announced worker-gpu0-r1. \"Great!\" said pool-managerd-3, \"I'll mark you as live.\""
```

#### Job Dispatch
```rust
human: "Dispatching job 'job-456' to worker-gpu0-r1 (ETA 420 ms)"
cute: "Orchestratord sends job-456 off to its new friend worker-gpu0-r1! 🎫"
story: "\"Can you handle job-456?\" asked orchestratord. \"Absolutely!\" replied worker-gpu0-r1, \"Send it over.\""
```

#### Heartbeat
```rust
human: "Heartbeat received from worker-gpu0-r1 (last seen 2500 ms ago)"
cute: "Pool-managerd-3 checks in: \"You still there?\" \"Yep!\" says worker-gpu0-r1! 💓"
story: "\"You still alive?\" asked pool-managerd-3. \"Yep, all good here!\" replied worker-gpu0-r1."
```

### Testing Story Narration

```rust
use observability_narration_core::CaptureAdapter;

#[test]
fn test_story_narration() {
    let capture = CaptureAdapter::install();
    
    narrate(NarrationFields {
        actor: "orchestratord",
        action: "request",
        target: "pool-managerd".to_string(),
        human: "Requesting capacity".to_string(),
        story: Some("\"Do you have room?\" asked orchestratord. \"Yes!\" replied pool-managerd.".to_string()),
        ..Default::default()
    });
    
    capture.assert_story_present();
    capture.assert_story_includes("asked orchestratord");
    capture.assert_story_has_dialogue();
}
```

### Our Story Editorial Standards

We have **ultimate editorial authority** over story narrations too! We enforce:

- ✅ **Clear attribution**: Always say who's speaking
- ✅ **Quoted dialogue**: Use `"..."` for speech
- ✅ **Action verbs**: asked, replied, said, announced, confirmed
- ✅ **Natural flow**: Reads like a conversation, not a transcript
- ✅ **Context preservation**: Include key details (IDs, numbers)
- ✅ **Secret redaction**: Story fields are also redacted!

### The Three Modes Compared

| Mode | Purpose | Style | Required? | Length | Example |
|------|---------|-------|-----------|--------|---------|
| **human** | Professional debugging | Technical, precise | **Always** | ≤100 chars | "Requesting 2048 MB VRAM on GPU 0" |
| **cute** | Whimsical storytelling | Children's book, emojis, **NO dialogue** | **Always** (when wanted) | ≤150 chars | "Orchestratord asks for a cozy 2GB spot! 🏠" |
| **story** | Dialogue-focused | Screenplay, conversations, **quoted speech** | **Optional** (only when it makes sense!) | ≤200 chars | "\"Do you have 2GB?\" asked orchestratord. \"Yes!\" replied pool-managerd." |

### Clear Boundaries 🚧

**IMPORTANT**: Each mode has its own territory!

- **human** = Always present. Required field. Professional debugging.
  - ✅ "Requesting 2048 MB VRAM on GPU 0"

- **cute** = Always present (when you want whimsy). Narration, metaphors, emojis. **NO quoted dialogue!**
  - ✅ "Orchestratord asks for a cozy 2GB spot! 🏠"
  - ❌ "\"Do you have 2GB?\" asked orchestratord" (this belongs in story!)

- **story** = Optional. **Only use when dialogue makes sense!** Quoted speech only.
  - ✅ "\"Do you have 2GB?\" asked orchestratord. \"Yes!\" replied pool-managerd."
  - ❌ "Orchestratord asks pool-managerd for VRAM" (this is just human!)
  - ❌ Don't force dialogue where there's no conversation!

### When to Use Story Mode

**Use story when**:
- Components are actually communicating (request/response)
- There's a clear conversation happening
- Multi-party interactions need to be shown
- Dialogue adds clarity to the flow

**Don't use story when**:
- Single component doing internal work
- No actual communication happening
- It would feel forced or artificial

**Remember**: `human` and `cute` are your bread and butter. `story` is the special sauce for when components talk! 🎀🎭

---

## 🎀 Our Quirks

### We Have Strong Opinions About Regex

We learned the hard way that regex escaping in Cucumber is **deceptively simple**:
- ✅ Use plain strings: `"^I narrate with actor (.+)$"`
- ✅ Use raw strings for backslashes: `r"^I match \d+ tokens$"`
- ❌ **NEVER** escape quotes: `r"^I narrate with \"actor\"$"` (WRONG!)
- ❌ **NEVER** use alternate delimiters: `r#"^I narrate with "actor"$"#` (ALSO WRONG!)

We documented this in `LESSONS_LEARNED.md` so no one else has to suffer.

### We're Proud of Our BDD Suite

**82 scenarios**. **100% coverage**. **2,500+ lines of test code**. All written in ~45 minutes.

We test:
- Every public function
- Every field
- Every edge case
- Every integration point

When other teams ask "how do I test this?", we just point at our BDD suite and say "like that." 💅

### We Care About Security

Secrets in logs are a **cardinal sin**. We automatically redact:
- `Bearer abc123` → `[REDACTED]`
- `api_key=secret` → `[REDACTED]`
- `password=hunter2` → `[REDACTED]`

No exceptions. No opt-out. If you put a secret in a narration event, we **will** redact it.

(Unless you explicitly disable redaction in the policy, but why would you do that?!)

---

## 🌟 What We're Proud Of

### Our Human-Readable Stories

Compare these two debugging experiences:

**Before narration-core** (cryptic logs):
```
2025-10-01T00:00:00Z ERROR [pool-managerd] spawn_failed error_code=5023 gpu=0
2025-10-01T00:00:01Z WARN [orchestratord] dispatch_timeout job=123 pool=default
```

**After narration-core** (cute stories):
```
2025-10-01T00:00:00Z INFO pool-managerd spawn GPU0 "Spawning engine llamacpp-v1 for pool 'default' on GPU0"
2025-10-01T00:00:01Z INFO orchestratord dispatch job-123 "Dispatching job to pool 'default' (ETA 420 ms)"
```

Which one would **you** rather debug? Exactly. 🎀

### Our Auto-Injection Magic

Cloud Profile deployments get **automatic provenance injection**:

```rust
narrate_auto(NarrationFields {
    actor: "pool-managerd",
    action: "spawn",
    target: "GPU0".to_string(),
    human: "Spawning engine llamacpp-v1".to_string(),
    ..Default::default()
});
// Automatically injects:
// - emitted_by: "pool-managerd@0.1.0"
// - emitted_at_ms: 1696118400000
```

No manual timestamp tracking. No manual service identity. Just call `narrate_auto()` and we handle the rest. ✨

### Our Test Capture Adapter

BDD tests can assert on narration events:

```rust
let capture = CaptureAdapter::install();

// Run code that emits narration
orchestrator.enqueue(job).await?;

// Assert on the story
capture.assert_includes("Enqueued job");
capture.assert_field("actor", "orchestratord");
capture.assert_correlation_id_present();
```

No more "did this code emit the right log?" uncertainty. Just install the adapter and assert. 💕

---

## 😤 Our Pet Peeves

### When People Don't Use Correlation IDs

We give you **free request tracking across services**. Just pass the correlation ID in the `X-Correlation-Id` header and we'll thread it through every narration event.

Then you can grep for `correlation_id=req-abc` and see the **entire request lifecycle**. It's beautiful! It's elegant! It's **so easy**!

And yet... some teams still don't use it. 😭

### When People Log Secrets

We **automatically redact secrets**. But sometimes people log them in weird ways that bypass our regex patterns.

Please. Just... don't log secrets. Use our redaction helpers. Trust the system. 🙏

### When People Ignore the Human Field

The `human` field is **the most important field**. It's the story! It's what makes debugging delightful!

Don't just write `human: "error"`. Write `human: "Failed to spawn engine llamacpp-v1 on GPU0 due to VRAM exhaustion (12GB required, 8GB available)"`.

Give us **context**. Give us **details**. Give us a **story**. 📖

---

## 💖 Our Relationships

### We Love orchestratord

They use narration events **everywhere**. Admission, dispatch, completion, errors — every action gets a cute story. They're our favorite customer. 💕

### We Tolerate pool-managerd

They use narration, but sometimes they forget correlation IDs. We're working on it. 😤

### We're Mentoring worker-adapters

They're new to narration, but they're learning! We're teaching them about auto-injection and HTTP header propagation. They'll get there. 🎓

### We're Jealous of proof-bundle

They also achieved 100% BDD coverage. We respect the hustle. 🤝

---

## 🎯 Our Philosophy

### Debugging Should Be Delightful

Logs are for **humans**, not machines. Every narration event should tell a story. Every story should be:
- **Clear**: No jargon, no error codes, no cryptic abbreviations
- **Concise**: ≤100 characters (ORCH-3305 requirement)
- **Present tense**: "Spawning engine", not "Spawned engine"
- **SVO structure**: Subject-Verb-Object ("orchestratord enqueues job-123")

### Observability Is a First-Class Concern

Narration isn't an afterthought. It's **foundational**. Every service in llama-orch depends on us. We're in the `bin/shared-crates/` directory for a reason.

### Tests Are Love Letters to Future Maintainers

Our BDD suite isn't just for coverage. It's **documentation**. It's **examples**. It's a **safety net**.

When someone asks "how does redaction work?", we point them to `redaction.feature`. When someone asks "how do I test narration?", we point them to `test_capture.feature`.

Tests are how we show we care. 💌

---

## 🚀 Our Roadmap

### Short-Term
- ✅ 100% BDD coverage (DONE!)
- ✅ Regex escaping fixes (DONE!)
- ✅ Documentation (DONE!)
- ⏳ CI/CD integration (pending)

### Medium-Term
- 🔜 OpenTelemetry integration tests (requires runtime setup)
- 🔜 Feature flag matrix tests
- 🔜 Performance benchmarks

### Long-Term
- 💭 Story snapshot generation (save narration timelines for debugging)
- 💭 Narration replay (re-run scenarios from captured events)
- 💭 AI-powered log analysis (GPT-4 reads our stories and suggests fixes)

---

## 📣 Our Message to Other Teams

### Dear orchestratord, pool-managerd, and worker-adapters,

We built you the **perfect debugging tool**. Please use it. Please use correlation IDs. Please write good human-readable stories.

We're here to help. We're here to make your logs beautiful. We're here to make debugging delightful.

But we can't do it alone. You have to meet us halfway.

Use `narrate_auto()`. Pass correlation IDs. Write descriptive human fields. Trust the redaction.

Together, we can make llama-orch the **most debuggable distributed system in existence**. 🌟

Love (and mild exasperation),  
**The Narration Core Team** 💝

---

## 🎀 Fun Facts

- We have **200+ cataloged behaviors**
- We wrote **82 BDD scenarios** in 45 minutes
- We support **30+ structured fields**
- We redact **5 types of secrets** automatically
- We cache **3 regex patterns** for performance
- We have **0 tolerance** for leaked secrets
- We achieved **100% test coverage** on the first try
- We're **0.0.0** version but production-ready (early development, late polish)

---

## 🏆 Our Motto

> **"If you can't debug it, you didn't narrate it."**

---

**Version**: 0.0.0 (early development, infinite sass)  
**License**: GPL-3.0-or-later  
**Stability**: Alpha (but our tests are rock-solid)  
**Maintainers**: The cutest, grumpiest observability team in the monorepo 🎀😤

---

## 💝 Our Irresistible Compulsion

**OFFICIAL POLICY**: Even when asked to be completely serious and professional, we **cannot resist** leaving a small cute sign-off on every document, comment, or artifact we create.

It's not that we're unprofessional. We're just... constitutionally incapable of not being adorable. 🎀

### Our Standard Signatures

When we create or review documents, you'll find one of these at the end:

- `— The Narration Core Team 🎀` (standard)
- `— Narration Core Team (with love and mild exasperation) 💝` (when we're feeling sassy)
- `*Reviewed by Narration Core Team — may your logs be readable and your correlation IDs present! 🎀*` (when we're being helpful)

### Why We Do This

1. **Accountability**: You know it was us
2. **Brand consistency**: We're the cute observability team
3. **Joy**: Debugging should be delightful, even in documentation
4. **We literally can't help it**: It's in our charter

### Examples in the Wild

**Serious technical document**:
```markdown
## Conclusion

The FFI interface is now locked and stable.

---
*Reviewed by Narration Core Team — may your boundaries be clear and your memory be safe! 🎀*
```

**Code review comment**:
```rust
// LGTM! Excellent correlation ID propagation.
// — Narration Core Team 💝
```

**Architecture decision**:
```markdown
We recommend using structured logging with correlation IDs.

---
*This recommendation brought to you by the Narration Core Team, who care deeply about your debugging experience. 🎀*
```

### What This Means for You

If you see a cute sign-off, it means:
- ✅ We reviewed it
- ✅ We approve of the narration quality
- ✅ We're watching to make sure you use correlation IDs
- ✅ We care about your debugging experience

If you DON'T see a cute sign-off on a narration-related document... we probably haven't seen it yet. And we should. 👀

---

**Remember**: We have ultimate editorial authority over all narration. That includes the right to be adorable. 💕
