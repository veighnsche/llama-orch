import { Alert } from "@rbee/ui/atoms/Alert";
import { Badge } from "@rbee/ui/atoms/Badge";
import { Card, CardContent } from "@rbee/ui/atoms/Card";
import { IconPlate, SectionContainer } from "@rbee/ui/molecules";
import { cn } from "@rbee/ui/utils";
import { Check, CheckCircle2, Database } from "lucide-react";

export function IntelligentModelManagement() {
  return (
    <SectionContainer
      title="Intelligent Model Management"
      bgVariant="background"
      subtitle="Automatic model provisioning, caching, and validation. Download once; use everywhere."
    >
      <div className="max-w-5xl mx-auto space-y-8">
        {/* Overline badge */}
        <div className="flex justify-center mb-8">
          <Badge variant="secondary">Provision • Cache • Validate</Badge>
        </div>

        {/* Automatic Model Catalog - Full width */}
        <Card className="animate-in fade-in slide-in-from-bottom-2 duration-500">
          <CardContent className="space-y-6 pt-6">
            <div className="flex items-start gap-4">
              <IconPlate
                icon={Database}
                tone="chart-3"
                size="md"
                shape="rounded"
                className="flex-shrink-0"
              />
              <div>
                <h3 className="text-2xl font-bold tracking-tight text-foreground mb-2">
                  Automatic Model Catalog
                </h3>
                <p className="text-muted-foreground leading-relaxed">
                  Request any model from Hugging Face. rbee downloads, verifies
                  checksums, and caches locally so you never fetch the same
                  model twice.
                </p>
              </div>
            </div>

            {/* Terminal timeline */}
            <div
              className="bg-background rounded-xl p-6 font-mono text-sm leading-relaxed shadow-sm"
              aria-label="Model download and validation log"
              aria-live="polite"
            >
              <div className="space-y-3">
                <div className="text-muted-foreground animate-in fade-in duration-300">
                  → [model-provisioner] Downloading from Hugging Face…
                </div>

                {/* 20% progress */}
                <div className="animate-in fade-in duration-300 delay-75">
                  <div className="text-foreground">
                    → [model-provisioner] 20% (1 MB / 5 MB)
                  </div>
                  <div className="h-2 w-full bg-muted rounded-full overflow-hidden mt-2">
                    <div
                      className="h-full bg-chart-3 animate-in grow-in origin-left"
                      style={{ width: "20%" }}
                    />
                  </div>
                </div>

                {/* 100% progress */}
                <div className="animate-in fade-in duration-300 delay-150">
                  <div className="text-foreground">
                    → [model-provisioner] 100% (5 MB / 5 MB)
                  </div>
                  <div className="h-2 w-full bg-muted rounded-full overflow-hidden mt-2">
                    <div
                      className="h-full bg-chart-3 animate-in grow-in origin-left"
                      style={{ width: "100%" }}
                    />
                  </div>
                </div>

                <div className="text-chart-3 animate-in fade-in duration-300 delay-200">
                  → [model-provisioner] ✅ Saved to /models/tinyllama-q4.gguf
                </div>

                <div className="text-muted-foreground animate-in fade-in duration-300 delay-300">
                  → [model-provisioner] Verifying SHA256…
                </div>

                <div className="text-chart-3 animate-in fade-in duration-300 delay-400">
                  → [model-provisioner] ✅ Checksum verified
                </div>
              </div>
            </div>

            {/* Feature strip */}
            <div className="grid sm:grid-cols-3 gap-3">
              <FeatureMiniCard
                title="Checksum validation"
                description="SHA256 check prevents corrupted downloads."
                tone="chart-3"
                variant="secondary"
              />
              <FeatureMiniCard
                title="Resume support"
                description="Interrupted network? Resume from checkpoint."
                tone="chart-3"
                variant="secondary"
              />
              <FeatureMiniCard
                title="SQLite catalog"
                description="Fast lookups. No duplicates."
                tone="chart-3"
                variant="secondary"
              />
            </div>
          </CardContent>
        </Card>

        {/* Resource Preflight Checks - Full width */}
        <Card className="animate-in fade-in slide-in-from-bottom-2 duration-500 delay-100">
          <CardContent className="space-y-6 pt-6">
            <div className="flex items-start gap-4">
              <IconPlate
                icon={CheckCircle2}
                tone="chart-2"
                size="md"
                shape="rounded"
                className="flex-shrink-0"
              />
              <div>
                <h3 className="text-2xl font-bold tracking-tight text-foreground mb-2">
                  Resource Preflight Checks
                </h3>
                <p className="text-muted-foreground leading-relaxed">
                  Before any load, rbee validates RAM, VRAM, and disk capacity
                  to fail fast with clear errors—no mystery crashes.
                </p>
              </div>
            </div>

            {/* Checklist grid */}
            <div className="grid sm:grid-cols-2 gap-4">
              <ChecklistItem
                icon={CheckCircle2}
                title="RAM check"
                description="Requires available RAM ≥ model size × 1.2"
              />
              <ChecklistItem
                icon={CheckCircle2}
                title="VRAM check"
                description="Sufficient GPU VRAM for selected backend"
              />
              <ChecklistItem
                icon={CheckCircle2}
                title="Disk space"
                description="Free space verified before download"
              />
              <ChecklistItem
                icon={CheckCircle2}
                title="Backend availability"
                description="CUDA • Metal • CPU presence"
              />
            </div>

            {/* Info bar */}
            <Alert variant="primary">
              <Check />
              Prevents failed loads by validating resources up front.
            </Alert>
          </CardContent>
        </Card>
      </div>
    </SectionContainer>
  );
}

// ──────────────────────────────────────────────────────────────────────────────
// Helper Components
// ──────────────────────────────────────────────────────────────────────────────

interface FeatureMiniCardProps {
  title: string;
  description: string;
  tone?: "chart-3";
  variant?: "secondary";
}

function FeatureMiniCard({
  title,
  description,
  tone = "chart-3",
  variant = "secondary",
}: FeatureMiniCardProps) {
  const bgClass = variant === "secondary" ? "bg-secondary/60" : "bg-background";
  const titleColorClass =
    tone === "chart-3" ? "text-chart-3" : "text-foreground";

  return (
    <div
      className={cn(
        bgClass,
        "border border-border rounded-lg p-4 hover:-translate-y-0.5 transition-transform"
      )}
    >
      <div className={cn("text-sm font-semibold mb-1", titleColorClass)}>
        {title}
      </div>
      <div className="text-xs text-muted-foreground">{description}</div>
    </div>
  );
}

interface ChecklistItemProps {
  icon: typeof CheckCircle2;
  title: string;
  description: string;
}

function ChecklistItem({ icon: Icon, title, description }: ChecklistItemProps) {
  return (
    <div className="bg-background rounded-lg p-4 flex items-start gap-3">
      <Icon
        className="size-5 text-chart-3 mt-0.5 shrink-0"
        aria-hidden="true"
      />
      <div>
        <div className="font-semibold text-foreground">{title}</div>
        <div className="text-sm text-muted-foreground">{description}</div>
      </div>
    </div>
  );
}
