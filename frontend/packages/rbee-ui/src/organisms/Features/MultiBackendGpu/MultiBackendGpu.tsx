import { Alert, AlertDescription, AlertTitle } from "@rbee/ui/atoms/Alert";
import { Badge } from "@rbee/ui/atoms/Badge";
import { Card, CardContent } from "@rbee/ui/atoms/Card";
import { IconCardHeader, SectionContainer, TerminalWindow } from "@rbee/ui/molecules";
import { cn } from "@rbee/ui/utils";
import { AlertTriangle, CheckCircle2, Cpu, X } from "lucide-react";

export function MultiBackendGpu() {
  return (
    <SectionContainer
      title="Multi-Backend GPU Support"
      bgVariant="background"
      subtitle="CUDA, Metal, and CPU backends with explicit device selection. No silent fallbacks—you control the hardware."
    >
      <div className="max-w-6xl mx-auto space-y-10">
        {/* 1. Policy Poster (full-bleed branded banner) */}
        <Card className="relative overflow-hidden border-primary/40 bg-gradient-to-b from-primary/10 to-background animate-in fade-in slide-in-from-bottom-2">
          <CardContent className="p-8 md:p-10 space-y-6">
            <IconCardHeader
              icon={AlertTriangle}
              title="GPU FAIL FAST policy"
              subtitle="No silent fallbacks. Clear errors with suggestions. You choose the backend."
              iconTone="primary"
              iconSize="md"
              titleClassName="text-3xl md:text-4xl font-extrabold"
              subtitleClassName="text-lg"
              useCardHeader={false}
            />

            {/* Prohibited pills (red) */}
            <div>
              <div className="text-sm font-semibold text-muted-foreground mb-2">
                Prohibited:
              </div>
              <div className="flex flex-wrap gap-2">
                <Badge variant="destructive">No GPU→CPU fallback</Badge>
                <Badge variant="destructive">No graceful degradation</Badge>
                <Badge variant="destructive">No implicit CPU reroute</Badge>
              </div>
            </div>

            {/* What happens pills (green) */}
            <div>
              <div className="text-sm font-semibold text-muted-foreground mb-2">
                What happens:
              </div>
              <div className="flex flex-wrap gap-2">
                <SuccessBadge>Fail fast (exit 1)</SuccessBadge>
                <SuccessBadge>Helpful error message</SuccessBadge>
                <SuccessBadge>Explicit backend selection</SuccessBadge>
              </div>
            </div>

            {/* Inline error alert */}
            <Alert variant="destructive">
              <X />
              <AlertTitle className="font-mono">
                Insufficient VRAM: need 4000 MB, have 2000 MB
              </AlertTitle>
              <AlertDescription>
                <ul className="list-disc pl-5 space-y-1">
                  <li>Use smaller quantized model (Q4_K_M instead of Q8_0)</li>
                  <li>Try CPU backend explicitly (--backend cpu)</li>
                  <li>Free VRAM by closing other applications</li>
                </ul>
              </AlertDescription>
            </Alert>
          </CardContent>
        </Card>

        {/* 2. Detection Console (wide terminal) */}
        <TerminalWindow
          showChrome
          title="rbee-hive detect — workstation.home.arpa"
          className="animate-in fade-in slide-in-from-bottom-2 delay-100"
        >
          <div className="text-chart-3">rbee-hive detect</div>
          <div className="mt-3 text-muted-foreground">Available backends:</div>
          <div className="mt-2 flex flex-wrap gap-2">
            <BackendBadge variant="primary">cuda × 2</BackendBadge>
            <BackendBadge variant="muted">cpu × 1</BackendBadge>
            <BackendBadge variant="success">metal × 0</BackendBadge>
          </div>
          <div className="mt-4 text-foreground">Total devices: 3</div>
          <div className="mt-4 pt-4 border-t border-border text-sm text-muted-foreground">
            Cached in the registry for fast lookups and policy routing.
          </div>
        </TerminalWindow>

        {/* 3. Microcards strip (3-up) */}
        <div className="grid sm:grid-cols-3 gap-3 animate-in fade-in slide-in-from-bottom-2 delay-150">
          <div className="bg-background rounded-xl border border-border p-4 flex items-start gap-3 hover:-translate-y-0.5 transition-transform">
            <Cpu className="size-5 shrink-0 mt-0.5 text-chart-2" aria-hidden="true" />
            <div>
              <div className="font-semibold text-foreground text-sm">Detection</div>
              <div className="text-xs text-muted-foreground mt-1">Scans CUDA, Metal, CPU and counts devices.</div>
            </div>
          </div>
          <div className="bg-background rounded-xl border border-border p-4 flex items-start gap-3 hover:-translate-y-0.5 transition-transform">
            <CheckCircle2 className="size-5 shrink-0 mt-0.5 text-primary" aria-hidden="true" />
            <div>
              <div className="font-semibold text-foreground text-sm">Explicit selection</div>
              <div className="text-xs text-muted-foreground mt-1">Choose backend & device—no surprises.</div>
            </div>
          </div>
          <div className="bg-background rounded-xl border border-border p-4 flex items-start gap-3 hover:-translate-y-0.5 transition-transform">
            <AlertTriangle className="size-5 shrink-0 mt-0.5 text-destructive" aria-hidden="true" />
            <div>
              <div className="font-semibold text-foreground text-sm">Helpful suggestions</div>
              <div className="text-xs text-muted-foreground mt-1">Actionable fixes on error.</div>
            </div>
          </div>
        </div>
      </div>
    </SectionContainer>
  );
}

// ──────────────────────────────────────────────────────────────────────────────
// Helper Components
// ──────────────────────────────────────────────────────────────────────────────

interface SuccessBadgeProps {
  children: React.ReactNode;
}

function SuccessBadge({ children }: SuccessBadgeProps) {
  return (
    <span className="inline-flex items-center gap-2 rounded-full bg-chart-3/10 text-chart-3 px-3 py-1 text-xs font-semibold">
      {children}
    </span>
  );
}

interface BackendBadgeProps {
  variant: "primary" | "muted" | "success";
  children: React.ReactNode;
}

function BackendBadge({ variant, children }: BackendBadgeProps) {
  const variantClasses = {
    primary: "bg-primary/10 text-primary",
    muted: "bg-muted text-foreground/80",
    success: "bg-emerald-500/10 text-emerald-400",
  };

  return (
    <span
      className={cn(
        "rounded-md px-2 py-1 text-xs font-semibold",
        variantClasses[variant]
      )}
    >
      {children}
    </span>
  );
}
