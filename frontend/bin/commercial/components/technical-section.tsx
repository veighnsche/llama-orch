import { Github } from "lucide-react"
import { Button } from "@/components/ui/button"

export function TechnicalSection() {
  return (
    <section className="py-24 bg-background">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance">
            Built by Engineers, for Engineers
          </h2>
        </div>

        <div className="grid md:grid-cols-2 gap-12 max-w-5xl mx-auto">
          {/* Architecture Highlights */}
          <div className="space-y-6">
            <h3 className="text-2xl font-bold text-foreground">Architecture Highlights</h3>
            <ul className="space-y-3">
              <li className="flex items-start gap-3">
                <div className="h-6 w-6 rounded-full bg-chart-3/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <div className="h-2 w-2 rounded-full bg-chart-3"></div>
                </div>
                <div>
                  <div className="font-medium text-foreground">BDD-Driven Development</div>
                  <div className="text-sm text-muted-foreground">42/62 scenarios passing (68% complete)</div>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <div className="h-6 w-6 rounded-full bg-chart-3/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <div className="h-2 w-2 rounded-full bg-chart-3"></div>
                </div>
                <div>
                  <div className="font-medium text-foreground">Cascading Shutdown Guarantee</div>
                  <div className="text-sm text-muted-foreground">No orphaned processes, clean VRAM lifecycle</div>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <div className="h-6 w-6 rounded-full bg-chart-3/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <div className="h-2 w-2 rounded-full bg-chart-3"></div>
                </div>
                <div>
                  <div className="font-medium text-foreground">Process Isolation</div>
                  <div className="text-sm text-muted-foreground">Clean VRAM lifecycle, no memory leaks</div>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <div className="h-6 w-6 rounded-full bg-chart-3/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <div className="h-2 w-2 rounded-full bg-chart-3"></div>
                </div>
                <div>
                  <div className="font-medium text-foreground">Protocol-Aware Orchestration</div>
                  <div className="text-sm text-muted-foreground">SSE, JSON, binary protocols supported</div>
                </div>
              </li>
              <li className="flex items-start gap-3">
                <div className="h-6 w-6 rounded-full bg-chart-3/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <div className="h-2 w-2 rounded-full bg-chart-3"></div>
                </div>
                <div>
                  <div className="font-medium text-foreground">Smart/Dumb Separation</div>
                  <div className="text-sm text-muted-foreground">Centralized intelligence, distributed execution</div>
                </div>
              </li>
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
      </div>
    </section>
  )
}
