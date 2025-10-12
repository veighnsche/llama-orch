import { Github } from "lucide-react"
import { Button } from '@/components/atoms/Button/Button'
import { SectionContainer, BulletListItem } from '@/components/molecules'

export function TechnicalSection() {
  return (
    <SectionContainer title="Built by Engineers, for Engineers">

      <div className="grid md:grid-cols-2 gap-12 max-w-5xl mx-auto">
        {/* Architecture Highlights */}
        <div className="space-y-6">
          <h3 className="text-2xl font-bold text-foreground">Architecture Highlights</h3>
          <ul className="space-y-3">
            <BulletListItem
              title="BDD-Driven Development"
              description="42/62 scenarios passing (68% complete)"
            />
            <BulletListItem
              title="Cascading Shutdown Guarantee"
              description="No orphaned processes, clean VRAM lifecycle"
            />
            <BulletListItem
              title="Process Isolation"
              description="Clean VRAM lifecycle, no memory leaks"
            />
            <BulletListItem
              title="Protocol-Aware Orchestration"
              description="SSE, JSON, binary protocols supported"
            />
            <BulletListItem
              title="Smart/Dumb Separation"
              description="Centralized intelligence, distributed execution"
            />
          </ul>
        </div>

        {/* Technology Stack */}
        <div className="space-y-6">
          <h3 className="text-2xl font-bold text-foreground">Technology Stack</h3>
          <div className="space-y-3">
            <div className="bg-muted border border-border rounded-lg p-4">
              <div className="font-medium text-foreground">Rust</div>
              <div className="text-sm text-muted-foreground">Performance + safety</div>
            </div>
            <div className="bg-muted border border-border rounded-lg p-4">
              <div className="font-medium text-foreground">Candle ML Framework</div>
              <div className="text-sm text-muted-foreground">Rust-native ML inference</div>
            </div>
            <div className="bg-muted border border-border rounded-lg p-4">
              <div className="font-medium text-foreground">Rhai Scripting</div>
              <div className="text-sm text-muted-foreground">Embedded, sandboxed scripting</div>
            </div>
            <div className="bg-muted border border-border rounded-lg p-4">
              <div className="font-medium text-foreground">SQLite</div>
              <div className="text-sm text-muted-foreground">Embedded database</div>
            </div>
            <div className="bg-muted border border-border rounded-lg p-4">
              <div className="font-medium text-foreground">Axum + Vue.js</div>
              <div className="text-sm text-muted-foreground">Async web framework + modern UI</div>
            </div>
          </div>

          <div className="bg-primary/10 border border-primary/30 rounded-lg p-4 flex items-center justify-between">
            <div>
              <div className="font-bold text-foreground">100% Open Source</div>
              <div className="text-sm text-muted-foreground">MIT License</div>
            </div>
            <Button variant="outline" size="sm" className="border-primary/30 bg-transparent">
              <Github className="h-4 w-4 mr-2" />
              View Source
            </Button>
          </div>
        </div>
      </div>
    </SectionContainer>
  )
}
