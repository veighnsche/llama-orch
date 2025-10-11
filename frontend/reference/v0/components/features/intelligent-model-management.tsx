import { Database, CheckCircle2 } from "lucide-react"

export function IntelligentModelManagement() {
  return (
    <section className="py-24 bg-white">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-slate-900 mb-6 text-balance">
            Intelligent Model Management
          </h2>
          <p className="text-xl text-slate-600 leading-relaxed">
            Automatic model provisioning, caching, and validation. Download once, use everywhere.
          </p>
        </div>

        <div className="max-w-5xl mx-auto space-y-8">
          {/* Model Catalog */}
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-8">
            <div className="flex items-start gap-4 mb-6">
              <div className="h-12 w-12 rounded-lg bg-green-100 flex items-center justify-center flex-shrink-0">
                <Database className="h-6 w-6 text-green-600" />
              </div>
              <div>
                <h3 className="text-2xl font-bold text-slate-900 mb-2">Automatic Model Catalog</h3>
                <p className="text-slate-600 leading-relaxed">
                  Request any model from Hugging Face. rbee downloads, validates checksums, and caches locally. Never
                  download the same model twice.
                </p>
              </div>
            </div>

            <div className="bg-slate-900 rounded-lg p-6 font-mono text-sm space-y-2">
              <div className="text-slate-400">→ [model-provisioner] 📦 Downloading model from Hugging Face</div>
              <div className="text-slate-300">→ [model-provisioner] Downloading... [████----] 20% (1 MB / 5 MB)</div>
              <div className="text-slate-300">→ [model-provisioner] Downloading... [████████] 100% (5 MB / 5 MB)</div>
              <div className="text-green-400">
                → [model-provisioner] ✅ Model downloaded to /models/tinyllama-q4.gguf
              </div>
              <div className="text-slate-400 mt-4">→ [model-provisioner] Verifying SHA256 checksum...</div>
              <div className="text-green-400">→ [model-provisioner] ✅ Checksum verified</div>
            </div>

            <div className="mt-6 grid md:grid-cols-3 gap-4">
              <div className="bg-white border border-slate-200 rounded-lg p-4">
                <div className="text-green-600 font-bold mb-1">Checksum Validation</div>
                <div className="text-slate-600 text-sm">SHA256 verification prevents corrupted downloads</div>
              </div>
              <div className="bg-white border border-slate-200 rounded-lg p-4">
                <div className="text-green-600 font-bold mb-1">Resume Support</div>
                <div className="text-slate-600 text-sm">Network interruptions? Resume from checkpoint</div>
              </div>
              <div className="bg-white border border-slate-200 rounded-lg p-4">
                <div className="text-green-600 font-bold mb-1">SQLite Catalog</div>
                <div className="text-slate-600 text-sm">Fast lookups, no duplicate downloads</div>
              </div>
            </div>
          </div>

          {/* Resource Preflight Checks */}
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-8">
            <div className="flex items-start gap-4 mb-6">
              <div className="h-12 w-12 rounded-lg bg-blue-100 flex items-center justify-center flex-shrink-0">
                <CheckCircle2 className="h-6 w-6 text-blue-600" />
              </div>
              <div>
                <h3 className="text-2xl font-bold text-slate-900 mb-2">Resource Preflight Checks</h3>
                <p className="text-slate-600 leading-relaxed">
                  Before loading any model, rbee validates RAM, VRAM, and disk space. Fail fast with clear error
                  messages instead of cryptic crashes.
                </p>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold text-slate-900">RAM Check</div>
                  <div className="text-slate-600 text-sm">
                    Validates available RAM ≥ model size × 1.2 before loading
                  </div>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold text-slate-900">VRAM Check</div>
                  <div className="text-slate-600 text-sm">Ensures GPU has sufficient VRAM for requested backend</div>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold text-slate-900">Disk Space Check</div>
                  <div className="text-slate-600 text-sm">Verifies free disk space before downloading models</div>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold text-slate-900">Backend Availability</div>
                  <div className="text-slate-600 text-sm">Confirms CUDA/Metal/CPU backend is installed</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
