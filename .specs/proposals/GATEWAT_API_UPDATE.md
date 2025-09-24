# 1) `control.yaml` — additive patch

```diff
@@
 openapi: 3.1.0
 info:
   title: llama-orch Control Plane
   version: 1.0.0
@@
 components:
+  parameters:
+    IdempotencyKey:
+      in: header
+      name: Idempotency-Key
+      required: false
+      schema: { type: string, maxLength: 128 }
+      description: Client-provided key to deduplicate retries of task admission.
+
+  responses:
+    ErrorResponse:
+      description: Error envelope
+      content:
+        application/json:
+          schema: { $ref: '#/components/schemas/ErrorEnvelope' }
+
   schemas:
+    ModelRef:
+      type: string
+      description: |
+        Canonical model reference string. Supported schemes are engine-dependent,
+        e.g.:
+          - hf:org/repo[:revision_or_tag]
+          - gguf:path/to/model.gguf
+          - s3://bucket/key
+          - file:/abs/path
+        Engines MUST document which schemes they accept.
@@
 paths:
   /v1/capabilities:
     get:
       summary: Capability discovery
@@
   /v1/sessions/{id}:
     get:
       summary: Get session info
       parameters:
         - in: path
           name: id
           required: true
           schema: { type: string, format: uuid }
       responses:
         '200':
           description: Session info
           content:
             application/json:
               schema: { $ref: '#/components/schemas/SessionInfo' }
+    post:
+      summary: Touch session (extend TTL)
+      operationId: SessionTouch
+      parameters:
+        - in: path
+          name: id
+          required: true
+          schema: { type: string, format: uuid }
+      responses:
+        '204': { description: TTL extended }
+        '404': { $ref: '#/components/responses/ErrorResponse' }
```

> Notes:
> • Adds `ModelRef` (glossary) for reuse in data plane.
> • Adds `Idempotency-Key` (reusable param).
> • Adds `ErrorResponse` (reusable for non-2xx).
> • Adds `POST /v1/sessions/{id}` (“touch” TTL).

---

# 2) `data.yaml` — additive patch

```diff
@@
 openapi: 3.1.0
 info:
   title: llama-orch Data Plane
   version: 1.0.0
@@
 components:
+  parameters:
+    IdempotencyKey:
+      in: header
+      name: Idempotency-Key
+      required: false
+      schema: { type: string, maxLength: 128 }
+      description: Client-provided key to deduplicate retries of task admission.
+
+  responses:
+    ErrorResponse:
+      description: Error envelope
+      content:
+        application/json:
+          schema: { $ref: '#/components/schemas/ErrorEnvelope' }
+
   schemas:
     TaskRequest:
       type: object
       properties:
-        model_ref: { type: string }
+        model_ref: { $ref: '#/components/schemas/ModelRef' }
@@
     ErrorEnvelope:
       type: object
       required: [ code, message ]
       properties:
         code: { $ref: '#/components/schemas/ErrorKind' }
         message: { type: string }
         engine: { $ref: '#/components/schemas/Engine', nullable: true }
+        policy_label: { type: string, nullable: true, description: Deterministic label naming the enforced policy path (e.g., queue.reject.rpm). }
         retriable: { type: boolean, nullable: true }
         retry_after_ms: { type: integer, format: int64, nullable: true }
@@
+    ModelRef:
+      type: string
+      description: |
+        Canonical model reference string. Supported schemes are engine-dependent,
+        e.g.:
+          - hf:org/repo[:revision_or_tag]
+          - gguf:path/to/model.gguf
+          - s3://bucket/key
+          - file:/abs/path
+        Engines MUST document which schemes they accept.
@@
 paths:
   /v1/tasks:
     post:
       summary: Admit task
+      parameters:
+        - $ref: '#/components/parameters/IdempotencyKey'
       requestBody:
         required: true
         content:
           application/json:
             schema: { $ref: '#/components/schemas/TaskRequest' }
       responses:
         '202':
           description: Accepted to queue
           content:
             application/json:
               schema: { $ref: '#/components/schemas/AdmissionResponse' }
-        '400': { description: Bad request }
-        '429': { description: Backpressure }
-        '503': { description: Pool unavailable }
+        '400': { $ref: '#/components/responses/ErrorResponse' }
+        '409': { $ref: '#/components/responses/ErrorResponse' }
+        '422': { $ref: '#/components/responses/ErrorResponse' }
+        '429': { $ref: '#/components/responses/ErrorResponse' }
+        '503': { $ref: '#/components/responses/ErrorResponse' }
@@
+  /v1/tasks/preview:
+    post:
+      summary: Validate a task request without enqueuing
+      requestBody:
+        required: true
+        content:
+          application/json:
+            schema: { $ref: '#/components/schemas/TaskRequest' }
+      responses:
+        '200':
+          description: Preview-only evaluation
+          content:
+            application/json:
+              schema:
+                type: object
+                properties:
+                  feasible: { type: boolean }
+                  reason: { type: string, nullable: true }
+                  predicted_start_ms: { type: integer, format: int64, nullable: true }
+        '400': { $ref: '#/components/responses/ErrorResponse' }
@@
   /v1/tasks/{id}/stream:
     get:
       summary: Stream events for a task
       parameters:
         - in: path
           name: id
           required: true
           schema: { type: string, format: uuid }
       responses:
         '200':
           description: SSE stream
+          headers:
+            X-Queue-Depth:
+              description: Queue depth observed at stream start
+              schema: { type: integer, format: int32 }
+            X-Budget-Tokens-Remaining:
+              description: Remaining token budget for the tenant/key (if quotas enabled)
+              schema: { type: integer, format: int64 }
           content:
             text/event-stream:
               schema:
                 type: string
               examples:
                 default:
                   summary: SSE events
                   value: |
                     : keep-alive
+                    id: 1
                     event: started
                     data: {"queue_position":2,"predicted_start_ms":1234}
                     event: token
                     data: {"t":"Hel","i":0}
                     event: metrics
                     data: {"queue_depth":2,"on_time_probability":0.92}
                     event: end
                     data: {"tokens_out":128,"decode_ms":9876}
-      x-sse-heartbeat: { interval_sec: 15 }
-      x-sse-resume: { supported: true, header: "Last-Event-ID" }
+      x-sse-heartbeat: { interval_sec: 15 }
+      x-sse-resume: { supported: true, header: "Last-Event-ID" }
@@
   /v1/tasks/{id}/cancel:
     post:
       summary: Cancel a running or queued task
       parameters:
         - in: path
           name: id
           required: true
           schema: { type: string, format: uuid }
       responses:
-        '200': { description: Cancelled }
+        '200':
+          description: Cancellation accepted; the stream MUST NOT emit further token events.
+        '404': { $ref: '#/components/responses/ErrorResponse' }
```

> Notes:
> • Adds idempotency header to `/v1/tasks`.
> • Adds **`/v1/tasks/preview`**.
> • Adds **headers** on stream start and **SSE heartbeats + event IDs** (via `x-sse-*` extensions).
> • Clarifies **cancel** contract.
> • Unifies **error envelopes** across 4xx/5xx.
> • Reuses the new `ModelRef` schema.

---

# 3) (Optional) New file — `.docs/GATEWAY_OPENAI.md`

```
# OpenAI-Compatible Gateway (Sidecar)

Status: Draft
Scope: Mapping between OpenAI-style requests and llama-orch data plane.

## Purpose
Offer client compatibility for existing OpenAI SDKs without changing the canonical llama-orch API.

## Endpoints

POST /v1/chat/completions  →  POST /v1/tasks
- Map `model` → `model_ref`
- Concatenate messages → `prompt` (or serialize to inputs)
- Map `max_tokens`, `temperature`, `top_p`, `seed` → TaskRequest fields
- Preserve `user`, `metadata` as labels if configured
- Return 202 AdmissionResponse { task_id, predicted_start_ms }

GET /v1/chat/completions/{task_id}/stream  →  GET /v1/tasks/{task_id}/stream
- Translate SSE frames:
  - `event: token` → chunk with `choices[0].delta.content`
  - `event: end`   → final message with usage
  - `event: error` → HTTP error with ErrorEnvelope → OpenAI-like error JSON

POST /v1/chat/completions/{task_id}/cancel → POST /v1/tasks/{task_id}/cancel

## Error Mapping
llama-orch ErrorEnvelope → OpenAI error object:
- `code` → `error.type` (string)
- `message` → `error.message`
- include `policy_label`, `retry_after_ms` when present

## Limits
- Tool calls/function calling are out of scope in v1 (handled at applet/toolkit layer).
- Streaming remains SSE (server-sent events).
```

---

## What I didn’t touch (but you can add anytime)

* **Examples folder**: `/examples/{task_request.json, error_envelope.json, sse_stream.txt}` — handy for SDK tests.
* **JSON Schema export**: generate JSON Schemas for `TaskRequest`, `ErrorEnvelope`, and SSE events — great for client-side validation.
