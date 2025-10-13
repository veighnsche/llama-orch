import { AlertTriangle, Cpu, CheckCircle2 } from 'lucide-react'
import { SectionContainer, IconBox } from '@/components/molecules'

export function MultiBackendGpu() {
  return (
    <SectionContainer
      title="Multi-Backend GPU Support"
      bgVariant="background"
      subtitle="CUDA, Metal, and CPU backends with explicit device selection. No silent fallbacks—you control the hardware."
    >
      <div className="max-w-6xl mx-auto space-y-10">
        {/* 1. Policy Poster (full-bleed branded banner) */}
        <div className="relative overflow-hidden rounded-2xl border border-primary/40 bg-gradient-to-b from-primary/10 to-background p-8 md:p-10 animate-in fade-in slide-in-from-bottom-2">
          <div className="flex items-start gap-4 mb-6">
            <IconBox icon={AlertTriangle} color="primary" size="md" className="flex-shrink-0" />
            <div>
              <h3 className="text-3xl md:text-4xl font-extrabold tracking-tight text-foreground mb-2">
                GPU FAIL FAST policy
              </h3>
              <p className="text-lg text-muted-foreground leading-relaxed">
                No silent fallbacks. Clear errors with suggestions. You choose the backend.
              </p>
            </div>
          </div>

          {/* Prohibited pills (red) */}
          <div className="mt-6">
            <div className="text-sm font-semibold text-muted-foreground mb-2">Prohibited:</div>
            <div className="flex flex-wrap gap-2">
              <span className="inline-flex items-center gap-2 rounded-full bg-destructive/10 text-destructive px-3 py-1 text-xs font-semibold">
                No GPU→CPU fallback
              </span>
              <span className="inline-flex items-center gap-2 rounded-full bg-destructive/10 text-destructive px-3 py-1 text-xs font-semibold">
                No graceful degradation
              </span>
              <span className="inline-flex items-center gap-2 rounded-full bg-destructive/10 text-destructive px-3 py-1 text-xs font-semibold">
                No implicit CPU reroute
              </span>
            </div>
          </div>

          {/* What happens pills (green) */}
          <div className="mt-4">
            <div className="text-sm font-semibold text-muted-foreground mb-2">What happens:</div>
            <div className="flex flex-wrap gap-2">
              <span className="inline-flex items-center gap-2 rounded-full bg-chart-3/10 text-chart-3 px-3 py-1 text-xs font-semibold">
                Fail fast (exit 1)
              </span>
              <span className="inline-flex items-center gap-2 rounded-full bg-chart-3/10 text-chart-3 px-3 py-1 text-xs font-semibold">
                Helpful error message
              </span>
              <span className="inline-flex items-center gap-2 rounded-full bg-chart-3/10 text-chart-3 px-3 py-1 text-xs font-semibold">
                Explicit backend selection
              </span>
            </div>
          </div>

          {/* Inline error toast */}
          <div className="mt-6 rounded-xl border border-destructive/40 bg-destructive/10 p-4 font-mono text-sm" role="alert">
            <div className="text-destructive font-semibold">❌ Insufficient VRAM: need 4000 MB, have 2000 MB</div>
            <ul className="mt-2 text-foreground list-disc pl-5 space-y-1">
              <li>Use smaller quantized model (Q4_K_M instead of Q8_0)</li>
              <li>Try CPU backend explicitly (--backend cpu)</li>
              <li>Free VRAM by closing other applications</li>
            </ul>
          </div>
        </div>

        {/* 2. Detection Console (wide terminal) */}
        <div className="rounded-2xl border border-border bg-card p-0 overflow-hidden animate-in fade-in slide-in-from-bottom-2 delay-100" role="region" aria-label="Backend detection results">
          {/* Faux terminal top bar */}
          <div className="flex items-center gap-1 bg-muted/50 px-4 py-2">
            <span className="size-2 rounded-full bg-red-500/70" aria-hidden="true" />
            <span className="size-2 rounded-full bg-yellow-500/70" aria-hidden="true" />
            <span className="size-2 rounded-full bg-green-500/70" aria-hidden="true" />
            <span className="ml-3 text-xs text-muted-foreground font-mono">rbee-hive detect — workstation.home.arpa</span>
          </div>

          {/* Log area */}
          <div className="bg-background p-6 font-mono text-sm leading-relaxed">
            <div className="text-chart-3">rbee-hive detect</div>
            <div className="mt-3 text-muted-foreground">Available backends:</div>
            <div className="mt-2 flex flex-wrap gap-2">
              <span className="rounded-md bg-primary/10 text-primary px-2 py-1 text-xs font-semibold">cuda × 2</span>
              <span className="rounded-md bg-muted text-foreground/80 px-2 py-1 text-xs font-semibold">cpu × 1</span>
              <span className="rounded-md bg-emerald-500/10 text-emerald-400 px-2 py-1 text-xs font-semibold">metal × 0</span>
            </div>
            <div className="mt-4 text-foreground">Total devices: 3</div>
          </div>

          {/* Benefit note */}
          <div className="px-6 py-3 border-t border-border text-sm text-muted-foreground">
            Cached in the registry for fast lookups and policy routing.
          </div>
        </div>

        {/* 3. Microcards strip (3-up) */}
        <div className="grid sm:grid-cols-3 gap-3 animate-in fade-in slide-in-from-bottom-2 delay-150">
          <div className="bg-background rounded-xl border border-border p-4 flex items-start gap-3 hover:-translate-y-0.5 transition-transform">
            <Cpu className="size-5 text-chart-2 shrink-0 mt-0.5" aria-hidden="true" />
            <div>
              <div className="font-semibold text-foreground text-sm">Detection</div>
              <div className="text-xs text-muted-foreground mt-1">Scans CUDA, Metal, CPU and counts devices.</div>
            </div>
          </div>

          <div className="bg-background rounded-xl border border-border p-4 flex items-start gap-3 hover:-translate-y-0.5 transition-transform">
            <CheckCircle2 className="size-5 text-primary shrink-0 mt-0.5" aria-hidden="true" />
            <div>
              <div className="font-semibold text-foreground text-sm">Explicit selection</div>
              <div className="text-xs text-muted-foreground mt-1">Choose backend & device—no surprises.</div>
            </div>
          </div>

          <div className="bg-background rounded-xl border border-border p-4 flex items-start gap-3 hover:-translate-y-0.5 transition-transform">
            <AlertTriangle className="size-5 text-destructive shrink-0 mt-0.5" aria-hidden="true" />
            <div>
              <div className="font-semibold text-foreground text-sm">Helpful suggestions</div>
              <div className="text-xs text-muted-foreground mt-1">Actionable fixes on error.</div>
            </div>
          </div>
        </div>
      </div>
    </SectionContainer>
  )
}
