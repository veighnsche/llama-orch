import { GitHubIcon } from '@/components/atoms/GitHubIcon/GitHubIcon'
import { Button } from '@/components/atoms/Button/Button'
import { SectionContainer, BulletListItem } from '@/components/molecules'
import Image from 'next/image'
import Link from 'next/link'

export function TechnicalSection() {
  return (
    <SectionContainer
      title="Built by Engineers, for Engineers"
      subtitle="Rust-native orchestrator with process isolation, protocol awareness, and policy routing via Rhai."
      headingId="tech-title"
      centered={true}
    >
      <div className="grid grid-cols-12 gap-6 lg:gap-10 max-w-6xl mx-auto motion-safe:animate-in motion-safe:fade-in-50 motion-safe:duration-500">
        {/* Left Column: Architecture Highlights + Diagram */}
        <div className="col-span-12 lg:col-span-6 space-y-6">
          {/* Architecture Highlights */}
          <div>
            <div className="text-xs tracking-wide uppercase text-muted-foreground mb-3">Core Principles</div>
            <h3 className="text-2xl font-bold text-foreground mb-4">Architecture Highlights</h3>
            <ul className="space-y-3">
              <BulletListItem
                title="BDD-Driven Development"
                description="42/62 scenarios passing (68% complete)"
                meta="Live CI coverage"
                variant="check"
              />
              <BulletListItem
                title="Cascading Shutdown Guarantee"
                description="No orphaned processes. Clean VRAM lifecycle."
                variant="check"
              />
              <BulletListItem
                title="Process Isolation"
                description="Worker-level sandboxes. Zero cross-leak."
                variant="check"
              />
              <BulletListItem
                title="Protocol-Aware Orchestration"
                description="SSE, JSON, binary protocols."
                variant="check"
              />
              <BulletListItem
                title="Smart/Dumb Separation"
                description="Central brain, distributed execution."
                variant="check"
              />
            </ul>

            {/* BDD Coverage Progress Bar */}
            <div className="mt-6">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-foreground">BDD Coverage</span>
                <span className="text-xs text-muted-foreground">42/62 scenarios passing</span>
              </div>
              <div className="relative h-2 rounded bg-muted">
                <div className="absolute inset-y-0 left-0 w-[68%] bg-chart-3 rounded" />
              </div>
              <p className="text-xs text-muted-foreground mt-1">68% complete</p>
            </div>
          </div>

          {/* Architecture Diagram (Desktop Only) */}
          <Image
            src="/images/rbee-arch.svg"
            width={920}
            height={560}
            className="hidden md:block rounded-2xl ring-1 ring-border/60 shadow-sm"
            alt="Professional systems architecture diagram of rbee orchestrator in dark theme (#0a0a0a background): Top section shows control plane with three interconnected hexagonal nodes - left hexagon labeled 'Rhai Policy Engine' (amber #f59e0b glow), center hexagon 'Request Scheduler' (teal #14b8a6 glow), right hexagon 'Health Monitor' (emerald #10b981 glow). Hexagons connected by bidirectional arrows with subtle pulse animation indicators. Middle section displays horizontal separator line with 'Protocol Layer' label, showing three protocol badges: 'SSE' (Server-Sent Events), 'JSON-RPC', and 'Binary' with corresponding icons. Bottom section illustrates worker pool: 4-6 rectangular worker nodes arranged in two rows, each labeled 'Worker Process' with GPU icon, VRAM gauge (showing 80-95% utilization in gradient fill), and model name tag (e.g., 'Llama-3-70B', 'Mistral-7B'). Each worker enclosed in dashed border indicating process isolation sandbox. Vertical arrows flow from protocol layer down to workers (request routing) and back up (response streaming). Right side shows cascading shutdown sequence: numbered steps (1→2→3) with graceful termination icons and 'Clean VRAM Release' annotation. Color scheme: dark charcoal background (#0a0a0a), white text (#fafafa), teal accent lines (#14b8a6) for active paths, amber (#f59e0b) for policy routing, muted gray (#71717a) for inactive elements. Typography: sans-serif (Inter or similar), medium weight for labels, semibold for component names. Subtle drop shadows on nodes for depth, minimal gradient overlays on worker gauges. Clean, technical aesthetic similar to AWS architecture diagrams or Kubernetes cluster visualizations. Landscape orientation optimized for desktop display, 16:10 aspect ratio, high contrast for readability, suitable for both light and dark UI contexts."
            priority
          />
        </div>

        {/* Right Column: Technology Stack (Sticky on Large Screens) */}
        <div className="col-span-12 lg:col-span-6 space-y-6 lg:sticky lg:top-20">
          <div>
            <div className="text-xs tracking-wide uppercase text-muted-foreground mb-3">Stack</div>
            <h3 className="text-2xl font-bold text-foreground mb-4">Technology Stack</h3>
            <div className="space-y-3">
              {/* Spec Cards */}
              <article
                role="group"
                aria-label="Tech: Rust"
                className="bg-muted/60 border border-border rounded-xl p-4 hover:border-primary/40 transition-colors motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-400"
              >
                <div className="font-semibold text-foreground">Rust</div>
                <div className="text-sm text-muted-foreground">Performance + memory safety.</div>
              </article>

              <article
                role="group"
                aria-label="Tech: Candle ML"
                className="bg-muted/60 border border-border rounded-xl p-4 hover:border-primary/40 transition-colors motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-400 motion-safe:delay-100"
              >
                <div className="font-semibold text-foreground">Candle ML</div>
                <div className="text-sm text-muted-foreground">Rust-native inference.</div>
              </article>

              <article
                role="group"
                aria-label="Tech: Rhai Scripting"
                className="bg-muted/60 border border-border rounded-xl p-4 hover:border-primary/40 transition-colors motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-400 motion-safe:delay-200"
              >
                <div className="font-semibold text-foreground">Rhai Scripting</div>
                <div className="text-sm text-muted-foreground">Embedded, sandboxed policies.</div>
              </article>

              <article
                role="group"
                aria-label="Tech: SQLite"
                className="bg-muted/60 border border-border rounded-xl p-4 hover:border-primary/40 transition-colors motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-400 motion-safe:delay-300"
              >
                <div className="font-semibold text-foreground">SQLite</div>
                <div className="text-sm text-muted-foreground">Embedded, zero-ops DB.</div>
              </article>

              <article
                role="group"
                aria-label="Tech: Axum + Vue.js"
                className="bg-muted/60 border border-border rounded-xl p-4 hover:border-primary/40 transition-colors motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-400 motion-safe:delay-400"
              >
                <div className="font-semibold text-foreground">Axum + Vue.js</div>
                <div className="text-sm text-muted-foreground">Async backend + modern UI.</div>
              </article>

              {/* Open Source CTA Card */}
              <article
                role="group"
                aria-label="Open Source Information"
                className="bg-primary/10 border border-primary/30 rounded-xl p-5 flex items-center justify-between motion-safe:animate-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-400 motion-safe:delay-500"
              >
                <div>
                  <div className="font-bold text-foreground">100% Open Source</div>
                  <div className="text-sm text-muted-foreground">MIT License</div>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  className="border-primary/30 bg-transparent"
                  aria-label="View rbee source on GitHub"
                  asChild
                >
                  <a href="https://github.com/yourusername/rbee" target="_blank" rel="noopener noreferrer">
                    <GitHubIcon className="h-4 w-4" />
                    View Source
                  </a>
                </Button>
              </article>

              {/* Architecture Docs Link */}
              <Link
                href="/docs/architecture"
                className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                Read Architecture →
              </Link>
            </div>
          </div>
        </div>
      </div>
    </SectionContainer>
  )
}
