import { Activity, XCircle, CheckCircle2 } from "lucide-react"
import { SectionContainer, IconBox } from "@/components/primitives"

export function RealTimeProgress() {
  return (
    <SectionContainer
      title="Real‑time Progress Tracking"
      bgVariant="background"
      subtitle="Live narration of every step. See model loading, token generation, and resource usage as it happens."
    >
            <div className="max-w-5xl mx-auto space-y-8">
          {/* SSE Narration */}
          <div className="bg-card border border-border rounded-lg p-8">
            <div className="flex items-start gap-4 mb-6">
              <IconBox icon={Activity} color="primary" size="lg" className="flex-shrink-0" />
              <div>
                <h3 className="text-2xl font-bold text-foreground mb-2">SSE Narration Architecture</h3>
                <p className="text-muted-foreground leading-relaxed">
                  Workers narrate their progress via Server-Sent Events. Every action—from model loading to token
                  generation—streams to your terminal in real‑time.
                </p>
              </div>
            </div>

            <div className="bg-background rounded-lg p-6 font-mono text-sm space-y-2">
              <div className="text-muted-foreground">→ [llm-worker-rbee] 🌅 Worker starting on port 8001</div>
              <div className="text-muted-foreground">→ [device-manager] 🖥️ Initialized CUDA device 1</div>
              <div className="text-muted-foreground">→ [model-loader] 📦 Loading model from /models/tinyllama-q4.gguf</div>
              <div className="text-chart-3">→ [model-loader] 🛏️ Model loaded! 669 MB cozy in VRAM!</div>
              <div className="text-muted-foreground">→ [http-server] 🚀 HTTP server ready on port 8001</div>
              <div className="text-muted-foreground mt-4">→ [candle-backend] 🚀 Starting inference (prompt: 18 chars)</div>
              <div className="text-muted-foreground">→ [tokenizer] 🍰 Tokenized prompt (4 tokens)</div>
              <div className="text-muted-foreground">→ [candle-backend] 🧹 Reset KV cache for fresh start</div>
              <div className="text-foreground mt-2">Once upon a time...</div>
              <div className="text-muted-foreground">→ [candle-backend] 🎯 Generated 10 tokens</div>
              <div className="text-chart-3 mt-2">
                → [candle-backend] 🎉 Inference complete! 20 tokens in 150ms (133 tok/s)
              </div>
            </div>

            <div className="mt-6 grid md:grid-cols-2 gap-4">
              <div className="bg-background rounded-lg p-4">
                <div className="text-primary font-bold mb-1">Tokens → stdout</div>
                <div className="text-muted-foreground text-sm">Generated text streams directly to your terminal</div>
              </div>
              <div className="bg-background rounded-lg p-4">
                <div className="text-primary font-bold mb-1">Narration → stderr</div>
                <div className="text-muted-foreground text-sm">Progress updates go to stderr for clean separation</div>
              </div>
            </div>
          </div>

          {/* Request Cancellation */}
          <div className="bg-card border border-border rounded-lg p-8">
            <div className="flex items-start gap-4 mb-6">
              <IconBox icon={XCircle} color="destructive" size="lg" className="flex-shrink-0" />
              <div>
                <h3 className="text-2xl font-bold text-foreground mb-2">Request Cancellation</h3>
                <p className="text-muted-foreground leading-relaxed">
                  Press Ctrl+C to cancel any request. Worker stops immediately, releases resources, and returns to idle
                  state. No orphaned processes.
                </p>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold text-foreground">Explicit Cancellation</div>
                  <div className="text-muted-foreground text-sm">POST /v1/cancel (idempotent)</div>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold text-foreground">Client Disconnect</div>
                  <div className="text-muted-foreground text-sm">Worker detects SSE stream closure within 1s</div>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold text-foreground">Immediate Cleanup</div>
                  <div className="text-muted-foreground text-sm">Stops token generation, releases slot, logs event</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </SectionContainer>
  )
}
