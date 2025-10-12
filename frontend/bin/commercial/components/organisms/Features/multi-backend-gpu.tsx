import { AlertTriangle, Cpu, CheckCircle2, XCircle } from 'lucide-react'
import { SectionContainer, IconBox } from '@/components/molecules'

export function MultiBackendGpu() {
  return (
    <SectionContainer
      title="Multi-Backend GPU Support"
      bgVariant="background"
      subtitle="CUDA, Metal, and CPU backends with explicit device selection. No silent fallbacks—you control the hardware."
    >
      <div className="max-w-5xl mx-auto space-y-8">
        {/* GPU FAIL FAST Policy */}
        <div className="bg-card border border-primary/50 rounded-lg p-8">
          <div className="flex items-start gap-4 mb-6">
            <IconBox icon={AlertTriangle} color="primary" size="lg" className="flex-shrink-0" />
            <div>
              <h3 className="text-2xl font-bold text-foreground mb-2">GPU FAIL FAST Policy</h3>
              <p className="text-muted-foreground leading-relaxed">
                No automatic fallbacks. No silent degradation. GPU fails? You get a clear error message with actionable
                suggestions. You decide what to do next.
              </p>
            </div>
          </div>

          <div className="bg-background rounded-lg p-6 space-y-4">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <XCircle className="h-4 w-4 text-destructive" />
                <span className="text-foreground font-mono text-sm">NO automatic backend fallback (GPU → CPU)</span>
              </div>
              <div className="flex items-center gap-2">
                <XCircle className="h-4 w-4 text-destructive" />
                <span className="text-foreground font-mono text-sm">NO graceful degradation</span>
              </div>
              <div className="flex items-center gap-2">
                <XCircle className="h-4 w-4 text-destructive" />
                <span className="text-foreground font-mono text-sm">NO CPU fallback on GPU failure</span>
              </div>
            </div>

            <div className="border-t border-border pt-4 space-y-2">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-chart-3" />
                <span className="text-foreground font-mono text-sm">FAIL FAST with exit code 1</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-chart-3" />
                <span className="text-foreground font-mono text-sm">Clear error message with suggestions</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-chart-3" />
                <span className="text-foreground font-mono text-sm">User explicitly chooses backend</span>
              </div>
            </div>
          </div>

          <div className="mt-6 bg-destructive/10 border border-destructive/50 rounded-lg p-4">
            <div className="font-mono text-sm space-y-1">
              <div className="text-destructive">❌ Insufficient VRAM: need 4000 MB, have 2000 MB</div>
              <div className="text-muted-foreground mt-2">Suggestions:</div>
              <div className="text-foreground pl-4">• Use smaller quantized model (Q4_K_M instead of Q8_0)</div>
              <div className="text-foreground pl-4">• Try CPU backend explicitly (--backend cpu)</div>
              <div className="text-foreground pl-4">• Free VRAM by closing other applications</div>
            </div>
          </div>
        </div>

        {/* Backend Detection */}
        <div className="bg-card border border-border rounded-lg p-8">
          <div className="flex items-start gap-4 mb-6">
            <IconBox icon={Cpu} color="chart-2" size="lg" className="flex-shrink-0" />
            <div>
              <h3 className="text-2xl font-bold text-foreground mb-2">Automatic Backend Detection</h3>
              <p className="text-muted-foreground leading-relaxed">
                rbee-hive detects all available backends and device counts on startup. Stored in registry for fast
                lookups.
              </p>
            </div>
          </div>

          <div className="bg-background rounded-lg p-6 font-mono text-sm space-y-2">
            <div className="text-muted-foreground"># On workstation.home.arpa</div>
            <div className="text-chart-3 mt-2">rbee-hive detect</div>
            <div className="text-foreground mt-4">Backend Detection Results:</div>
            <div className="text-foreground">==========================</div>
            <div className="text-foreground mt-2">Available backends: 2</div>
            <div className="text-foreground pl-4">- cpu: 1 device(s)</div>
            <div className="text-foreground pl-4">- cuda: 2 device(s)</div>
            <div className="text-foreground mt-2">Total devices: 3</div>
          </div>
        </div>
      </div>
    </SectionContainer>
  )
}
