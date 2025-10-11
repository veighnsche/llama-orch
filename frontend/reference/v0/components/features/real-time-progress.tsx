import { Activity, XCircle, CheckCircle2 } from "lucide-react"

export function RealTimeProgress() {
  return (
    <section className="py-24 bg-slate-900">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-white mb-6 text-balance">Real-time Progress Tracking</h2>
          <p className="text-xl text-slate-300 leading-relaxed">
            Live narration of every step. See model loading, token generation, and resource usage as it happens.
          </p>
        </div>

        <div className="max-w-5xl mx-auto space-y-8">
          {/* SSE Narration */}
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-8">
            <div className="flex items-start gap-4 mb-6">
              <div className="h-12 w-12 rounded-lg bg-amber-500/10 flex items-center justify-center flex-shrink-0">
                <Activity className="h-6 w-6 text-amber-500" />
              </div>
              <div>
                <h3 className="text-2xl font-bold text-white mb-2">SSE Narration Architecture</h3>
                <p className="text-slate-300 leading-relaxed">
                  Workers narrate their progress via Server-Sent Events. Every action—from model loading to token
                  generation—streams to your terminal in real-time.
                </p>
              </div>
            </div>

            <div className="bg-slate-950 rounded-lg p-6 font-mono text-sm space-y-2">
              <div className="text-slate-400">→ [llm-worker-rbee] 🌅 Worker starting on port 8001</div>
              <div className="text-slate-400">→ [device-manager] 🖥️ Initialized CUDA device 1</div>
              <div className="text-slate-400">→ [model-loader] 📦 Loading model from /models/tinyllama-q4.gguf</div>
              <div className="text-green-400">→ [model-loader] 🛏️ Model loaded! 669 MB cozy in VRAM!</div>
              <div className="text-slate-400">→ [http-server] 🚀 HTTP server ready on port 8001</div>
              <div className="text-slate-400 mt-4">→ [candle-backend] 🚀 Starting inference (prompt: 18 chars)</div>
              <div className="text-slate-400">→ [tokenizer] 🍰 Tokenized prompt (4 tokens)</div>
              <div className="text-slate-400">→ [candle-backend] 🧹 Reset KV cache for fresh start</div>
              <div className="text-slate-300 mt-2">Once upon a time...</div>
              <div className="text-slate-400">→ [candle-backend] 🎯 Generated 10 tokens</div>
              <div className="text-green-400 mt-2">
                → [candle-backend] 🎉 Inference complete! 20 tokens in 150ms (133 tok/s)
              </div>
            </div>

            <div className="mt-6 grid md:grid-cols-2 gap-4">
              <div className="bg-slate-950 rounded-lg p-4">
                <div className="text-amber-500 font-bold mb-1">Tokens → stdout</div>
                <div className="text-slate-400 text-sm">Generated text streams directly to your terminal</div>
              </div>
              <div className="bg-slate-950 rounded-lg p-4">
                <div className="text-amber-500 font-bold mb-1">Narration → stderr</div>
                <div className="text-slate-400 text-sm">Progress updates go to stderr for clean separation</div>
              </div>
            </div>
          </div>

          {/* Request Cancellation */}
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-8">
            <div className="flex items-start gap-4 mb-6">
              <div className="h-12 w-12 rounded-lg bg-red-500/10 flex items-center justify-center flex-shrink-0">
                <XCircle className="h-6 w-6 text-red-500" />
              </div>
              <div>
                <h3 className="text-2xl font-bold text-white mb-2">Request Cancellation</h3>
                <p className="text-slate-300 leading-relaxed">
                  Press Ctrl+C to cancel any request. Worker stops immediately, releases resources, and returns to idle
                  state. No orphaned processes.
                </p>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 text-green-500 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold text-white">Explicit Cancellation</div>
                  <div className="text-slate-400 text-sm">DELETE /v1/inference/&lt;request_id&gt; (idempotent)</div>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 text-green-500 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold text-white">Client Disconnect</div>
                  <div className="text-slate-400 text-sm">Worker detects SSE stream closure within 1s</div>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 text-green-500 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold text-white">Immediate Cleanup</div>
                  <div className="text-slate-400 text-sm">Stops token generation, releases slot, logs event</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
