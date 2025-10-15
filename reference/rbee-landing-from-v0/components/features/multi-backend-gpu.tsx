import { AlertTriangle, Cpu, CheckCircle2, XCircle } from "lucide-react"

export function MultiBackendGpu() {
  return (
    <section className="py-24 bg-slate-900">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-white mb-6 text-balance">Multi-Backend GPU Support</h2>
          <p className="text-xl text-slate-300 leading-relaxed">
            CUDA, Metal, and CPU backends with explicit device selection. No silent fallbacks—you control the hardware.
          </p>
        </div>

        <div className="max-w-5xl mx-auto space-y-8">
          {/* GPU FAIL FAST Policy */}
          <div className="bg-slate-800 border border-amber-500/50 rounded-lg p-8">
            <div className="flex items-start gap-4 mb-6">
              <div className="h-12 w-12 rounded-lg bg-amber-500/10 flex items-center justify-center flex-shrink-0">
                <AlertTriangle className="h-6 w-6 text-amber-500" />
              </div>
              <div>
                <h3 className="text-2xl font-bold text-white mb-2">GPU FAIL FAST Policy</h3>
                <p className="text-slate-300 leading-relaxed">
                  No automatic fallbacks. No silent degradation. GPU fails? You get a clear error message with
                  actionable suggestions. You decide what to do next.
                </p>
              </div>
            </div>

            <div className="bg-slate-950 rounded-lg p-6 space-y-4">
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <XCircle className="h-4 w-4 text-red-500" />
                  <span className="text-slate-300 font-mono text-sm">NO automatic backend fallback (GPU → CPU)</span>
                </div>
                <div className="flex items-center gap-2">
                  <XCircle className="h-4 w-4 text-red-500" />
                  <span className="text-slate-300 font-mono text-sm">NO graceful degradation</span>
                </div>
                <div className="flex items-center gap-2">
                  <XCircle className="h-4 w-4 text-red-500" />
                  <span className="text-slate-300 font-mono text-sm">NO CPU fallback on GPU failure</span>
                </div>
              </div>

              <div className="border-t border-slate-700 pt-4 space-y-2">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-500" />
                  <span className="text-slate-300 font-mono text-sm">FAIL FAST with exit code 1</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-500" />
                  <span className="text-slate-300 font-mono text-sm">Clear error message with suggestions</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-500" />
                  <span className="text-slate-300 font-mono text-sm">User explicitly chooses backend</span>
                </div>
              </div>
            </div>

            <div className="mt-6 bg-red-950/50 border border-red-500/50 rounded-lg p-4">
              <div className="font-mono text-sm space-y-1">
                <div className="text-red-400">❌ Insufficient VRAM: need 4000 MB, have 2000 MB</div>
                <div className="text-slate-400 mt-2">Suggestions:</div>
                <div className="text-slate-300 pl-4">• Use smaller quantized model (Q4_K_M instead of Q8_0)</div>
                <div className="text-slate-300 pl-4">• Try CPU backend explicitly (--backend cpu)</div>
                <div className="text-slate-300 pl-4">• Free VRAM by closing other applications</div>
              </div>
            </div>
          </div>

          {/* Backend Detection */}
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-8">
            <div className="flex items-start gap-4 mb-6">
              <div className="h-12 w-12 rounded-lg bg-blue-500/10 flex items-center justify-center flex-shrink-0">
                <Cpu className="h-6 w-6 text-blue-500" />
              </div>
              <div>
                <h3 className="text-2xl font-bold text-white mb-2">Automatic Backend Detection</h3>
                <p className="text-slate-300 leading-relaxed">
                  rbee-hive detects all available backends and device counts on startup. Stored in registry for fast
                  lookups.
                </p>
              </div>
            </div>

            <div className="bg-slate-950 rounded-lg p-6 font-mono text-sm space-y-2">
              <div className="text-slate-400"># On workstation.home.arpa</div>
              <div className="text-green-400 mt-2">rbee-hive detect</div>
              <div className="text-slate-300 mt-4">Backend Detection Results:</div>
              <div className="text-slate-300">==========================</div>
              <div className="text-slate-300 mt-2">Available backends: 2</div>
              <div className="text-slate-300 pl-4">- cpu: 1 device(s)</div>
              <div className="text-slate-300 pl-4">- cuda: 2 device(s)</div>
              <div className="text-slate-300 mt-2">Total devices: 3</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
