import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/atoms/Tabs/Tabs'
import { Code, Cpu, Gauge, Zap } from 'lucide-react'
import { SectionContainer, CodeBlock, BenefitCallout } from '@/components/molecules'

export function FeaturesSection() {
  return (
    <SectionContainer title="Enterprise-Grade Features. Homelab Simplicity." bgVariant="secondary">
      <div className="max-w-5xl mx-auto">
        <Tabs defaultValue="api" className="w-full">
          <TabsList className="grid w-full grid-cols-2 lg:grid-cols-4 h-auto">
            <TabsTrigger value="api" className="flex items-center gap-2 py-3">
              <Code className="h-4 w-4" />
              <span className="hidden sm:inline">OpenAI-Compatible</span>
              <span className="sm:hidden">API</span>
            </TabsTrigger>
            <TabsTrigger value="gpu" className="flex items-center gap-2 py-3">
              <Cpu className="h-4 w-4" />
              <span className="hidden sm:inline">Multi-GPU</span>
              <span className="sm:hidden">GPU</span>
            </TabsTrigger>
            <TabsTrigger value="scheduler" className="flex items-center gap-2 py-3">
              <Gauge className="h-4 w-4" />
              <span className="hidden sm:inline">Scheduler</span>
              <span className="sm:hidden">Rhai</span>
            </TabsTrigger>
            <TabsTrigger value="sse" className="flex items-center gap-2 py-3">
              <Zap className="h-4 w-4" />
              <span className="hidden sm:inline">Real‑time</span>
              <span className="sm:hidden">SSE</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="api" className="mt-8">
            <div className="bg-card border border-border rounded-lg p-8 space-y-6">
              <div>
                <h3 className="text-2xl font-bold text-card-foreground mb-3">OpenAI-Compatible API</h3>
                <p className="text-muted-foreground leading-relaxed">
                  Drop-in replacement for OpenAI API. Works with Zed, Cursor, Continue, and any tool that supports
                  OpenAI.
                </p>
              </div>

              <CodeBlock
                language="bash"
                code={`# Before: Using OpenAI
export OPENAI_API_KEY=sk-...

# After: Using rbee (same code!)
export OPENAI_API_BASE=http://localhost:8080/v1`}
              />

              <BenefitCallout variant="success" text="No code changes. Just point to localhost." />
            </div>
          </TabsContent>

          <TabsContent value="gpu" className="mt-8">
            <div className="bg-card border border-border rounded-lg p-8 space-y-6">
              <div>
                <h3 className="text-2xl font-bold text-card-foreground mb-3">Multi-GPU Orchestration</h3>
                <p className="text-muted-foreground leading-relaxed">
                  Orchestrate across the CUDA, Metal, and CPU backends you configure. Use every GPU you own.
                </p>
              </div>

              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <div className="flex-shrink-0 w-32 text-sm text-muted-foreground">RTX 4090 #1</div>
                  <div className="flex-1 h-8 bg-muted rounded-full overflow-hidden">
                    <div className="h-full bg-primary flex items-center justify-end pr-2" style={{ width: '92%' }}>
                      <span className="text-xs text-primary-foreground font-medium">92%</span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="flex-shrink-0 w-32 text-sm text-muted-foreground">RTX 4090 #2</div>
                  <div className="flex-1 h-8 bg-muted rounded-full overflow-hidden">
                    <div className="h-full bg-primary flex items-center justify-end pr-2" style={{ width: '88%' }}>
                      <span className="text-xs text-primary-foreground font-medium">88%</span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="flex-shrink-0 w-32 text-sm text-muted-foreground">M2 Ultra</div>
                  <div className="flex-1 h-8 bg-muted rounded-full overflow-hidden">
                    <div className="h-full bg-primary flex items-center justify-end pr-2" style={{ width: '76%' }}>
                      <span className="text-xs text-primary-foreground font-medium">76%</span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="flex-shrink-0 w-32 text-sm text-muted-foreground">CPU Backend</div>
                  <div className="flex-1 h-8 bg-muted rounded-full overflow-hidden">
                    <div className="h-full bg-chart-2 flex items-center justify-end pr-2" style={{ width: '34%' }}>
                      <span className="text-xs text-primary-foreground font-medium">34%</span>
                    </div>
                  </div>
                </div>
              </div>

              <BenefitCallout variant="info" text="10x throughput by using all your hardware." />
            </div>
          </TabsContent>

          <TabsContent value="scheduler" className="mt-8">
            <div className="bg-card border border-border rounded-lg p-8 space-y-6">
              <div>
                <h3 className="text-2xl font-bold text-card-foreground mb-3">Programmable Rhai Scheduler</h3>
                <p className="text-muted-foreground leading-relaxed">
                  Write custom routing logic in Rhai. Route large models to multi-GPU setups, image generation to CUDA,
                  everything else to cheapest.
                </p>
              </div>

              <CodeBlock
                language="rust"
                code={`// Custom routing logic
if task.model.contains("70b") {
  route_to("multi-gpu-cluster")
} else if task.type == "image" {
  route_to("cuda-only")
} else {
  route_to("cheapest")
}`}
              />

              <BenefitCallout variant="primary" text="Optimize for cost, latency, or compliance—your rules." />
            </div>
          </TabsContent>

          <TabsContent value="sse" className="mt-8">
            <div className="bg-card border border-border rounded-lg p-8 space-y-6">
              <div>
                <h3 className="text-2xl font-bold text-card-foreground mb-3">Task-Based API with SSE</h3>
                <p className="text-muted-foreground leading-relaxed">
                  Real‑time progress updates. See model loading, token generation, and cost tracking as it happens.
                </p>
              </div>

              <CodeBlock
                language="json"
                code={`→ event: task.created
{ "id": "task_123", "status": "pending" }

→ event: model.loading
{ "progress": 0.45, "eta": "2.1s" }

→ event: token.generated
{ "token": "const", "total": 1 }

→ event: token.generated
{ "token": " api", "total": 2 }`}
              />

              <BenefitCallout text="Full visibility into every inference job." />
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </SectionContainer>
  )
}
