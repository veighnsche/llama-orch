import { Database, CheckCircle2 } from "lucide-react"
import { SectionContainer, IconBox } from '@/components/molecules'

export function IntelligentModelManagement() {
  return (
    <SectionContainer
      title="Intelligent Model Management"
      bgVariant="background"
      subtitle="Automatic model provisioning, caching, and validation. Download once, use everywhere."
    >
            <div className="max-w-5xl mx-auto space-y-8">
          {/* Model Catalog */}
          <div className="bg-card border border-border rounded-lg p-8">
            <div className="flex items-start gap-4 mb-6">
              <IconBox icon={Database} color="chart-3" size="lg" className="flex-shrink-0" />
              <div>
                <h3 className="text-2xl font-bold text-foreground mb-2">Automatic Model Catalog</h3>
                <p className="text-muted-foreground leading-relaxed">
                  Request any model from Hugging Face. rbee downloads, validates checksums, and caches locally. Never
                  download the same model twice.
                </p>
              </div>
            </div>

            <div className="bg-background rounded-lg p-6 font-mono text-sm space-y-2">
              <div className="text-muted-foreground">→ [model-provisioner] 📦 Downloading model from Hugging Face</div>
              <div className="text-foreground">→ [model-provisioner] Downloading... [████----] 20% (1 MB / 5 MB)</div>
              <div className="text-foreground">→ [model-provisioner] Downloading... [████████] 100% (5 MB / 5 MB)</div>
              <div className="text-chart-3">
                → [model-provisioner] ✅ Model downloaded to /models/tinyllama-q4.gguf
              </div>
              <div className="text-muted-foreground mt-4">→ [model-provisioner] Verifying SHA256 checksum...</div>
              <div className="text-chart-3">→ [model-provisioner] ✅ Checksum verified</div>
            </div>

            <div className="mt-6 grid md:grid-cols-3 gap-4">
              <div className="bg-secondary border border-border rounded-lg p-4">
                <div className="text-chart-3 font-bold mb-1">Checksum Validation</div>
                <div className="text-muted-foreground text-sm">SHA256 verification prevents corrupted downloads</div>
              </div>
              <div className="bg-secondary border border-border rounded-lg p-4">
                <div className="text-chart-3 font-bold mb-1">Resume Support</div>
                <div className="text-muted-foreground text-sm">Network interruptions? Resume from checkpoint</div>
              </div>
              <div className="bg-secondary border border-border rounded-lg p-4">
                <div className="text-chart-3 font-bold mb-1">SQLite Catalog</div>
                <div className="text-muted-foreground text-sm">Fast lookups, no duplicate downloads</div>
              </div>
            </div>
          </div>

          {/* Resource Preflight Checks */}
          <div className="bg-card border border-border rounded-lg p-8">
            <div className="flex items-start gap-4 mb-6">
              <IconBox icon={CheckCircle2} color="chart-2" size="lg" className="flex-shrink-0" />
              <div>
                <h3 className="text-2xl font-bold text-foreground mb-2">Resource Preflight Checks</h3>
                <p className="text-muted-foreground leading-relaxed">
                  Before loading any model, rbee validates RAM, VRAM, and disk space. Fail fast with clear error
                  messages instead of cryptic crashes.
                </p>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold text-foreground">RAM Check</div>
                  <div className="text-muted-foreground text-sm">
                    Validates available RAM ≥ model size × 1.2 before loading
                  </div>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold text-foreground">VRAM Check</div>
                  <div className="text-muted-foreground text-sm">Ensures GPU has sufficient VRAM for requested backend</div>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold text-foreground">Disk Space Check</div>
                  <div className="text-muted-foreground text-sm">Verifies free disk space before downloading models</div>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold text-foreground">Backend Availability</div>
                  <div className="text-muted-foreground text-sm">Confirms CUDA/Metal/CPU backend is installed</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </SectionContainer>
  )
}
