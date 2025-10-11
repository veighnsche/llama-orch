<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-008: Implemented RealTimeProgress -->
<script setup lang="ts">
import { Activity, XCircle, CheckCircle2 } from 'lucide-vue-next'
</script>

<template>
  <section class="py-24 bg-background">
    <div class="container mx-auto px-4">
      <div class="max-w-4xl mx-auto text-center mb-16">
        <h2 class="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance">Real-time Progress Tracking</h2>
        <p class="text-xl text-muted-foreground leading-relaxed">
          Live narration of every step. See model loading, token generation, and resource usage as it happens.
        </p>
      </div>

      <div class="max-w-5xl mx-auto space-y-8">
        <!-- SSE Narration -->
        <div class="bg-card border border-border rounded-lg p-8">
          <div class="flex items-start gap-4 mb-6">
            <div class="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
              <Activity class="h-6 w-6 text-primary" />
            </div>
            <div>
              <h3 class="text-2xl font-bold text-foreground mb-2">SSE Narration Architecture</h3>
              <p class="text-muted-foreground leading-relaxed">
                Workers narrate their progress via Server-Sent Events. Every action—from model loading to token
                generation—streams to your terminal in real-time.
              </p>
            </div>
          </div>

          <div class="bg-secondary rounded-lg p-6 font-mono text-sm space-y-2">
            <div class="text-muted-foreground">→ [llm-worker-rbee] 🌅 Worker starting on port 8001</div>
            <div class="text-muted-foreground">→ [device-manager] 🖥️ Initialized CUDA device 1</div>
            <div class="text-muted-foreground">→ [model-loader] 📦 Loading model from /models/tinyllama-q4.gguf</div>
            <div class="text-accent">→ [model-loader] 🛏️ Model loaded! 669 MB cozy in VRAM!</div>
            <div class="text-muted-foreground">→ [http-server] 🚀 HTTP server ready on port 8001</div>
            <div class="text-muted-foreground mt-4">→ [candle-backend] 🚀 Starting inference (prompt: 18 chars)</div>
            <div class="text-muted-foreground">→ [tokenizer] 🍰 Tokenized prompt (4 tokens)</div>
            <div class="text-muted-foreground">→ [candle-backend] 🧹 Reset KV cache for fresh start</div>
            <div class="text-foreground mt-2">Once upon a time...</div>
            <div class="text-muted-foreground">→ [candle-backend] 🎯 Generated 10 tokens</div>
            <div class="text-accent mt-2">
              → [candle-backend] 🎉 Inference complete! 20 tokens in 150ms (133 tok/s)
            </div>
          </div>

          <div class="mt-6 grid md:grid-cols-2 gap-4">
            <div class="bg-secondary rounded-lg p-4">
              <div class="text-primary font-bold mb-1">Tokens → stdout</div>
              <div class="text-muted-foreground text-sm">Generated text streams directly to your terminal</div>
            </div>
            <div class="bg-secondary rounded-lg p-4">
              <div class="text-primary font-bold mb-1">Narration → stderr</div>
              <div class="text-muted-foreground text-sm">Progress updates go to stderr for clean separation</div>
            </div>
          </div>
        </div>

        <!-- Request Cancellation -->
        <div class="bg-card border border-border rounded-lg p-8">
          <div class="flex items-start gap-4 mb-6">
            <div class="h-12 w-12 rounded-lg bg-destructive/10 flex items-center justify-center flex-shrink-0">
              <XCircle class="h-6 w-6 text-destructive" />
            </div>
            <div>
              <h3 class="text-2xl font-bold text-foreground mb-2">Request Cancellation</h3>
              <p class="text-muted-foreground leading-relaxed">
                Press Ctrl+C to cancel any request. Worker stops immediately, releases resources, and returns to idle
                state. No orphaned processes.
              </p>
            </div>
          </div>

          <div class="space-y-4">
            <div class="flex items-start gap-3">
              <CheckCircle2 class="h-5 w-5 text-accent flex-shrink-0 mt-0.5" />
              <div>
                <div class="font-bold text-foreground">Explicit Cancellation</div>
                <div class="text-muted-foreground text-sm">DELETE /v1/inference/&lt;request_id&gt; (idempotent)</div>
              </div>
            </div>
            <div class="flex items-start gap-3">
              <CheckCircle2 class="h-5 w-5 text-accent flex-shrink-0 mt-0.5" />
              <div>
                <div class="font-bold text-foreground">Client Disconnect</div>
                <div class="text-muted-foreground text-sm">Worker detects SSE stream closure within 1s</div>
              </div>
            </div>
            <div class="flex items-start gap-3">
              <CheckCircle2 class="h-5 w-5 text-accent flex-shrink-0 mt-0.5" />
              <div>
                <div class="font-bold text-foreground">Immediate Cleanup</div>
                <div class="text-muted-foreground text-sm">Stops token generation, releases slot, logs event</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
