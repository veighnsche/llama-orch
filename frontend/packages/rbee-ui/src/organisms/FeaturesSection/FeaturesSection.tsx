'use client'

import { Tabs, TabsContent, TabsList, TabsTrigger } from '@rbee/ui/atoms/Tabs'
import { Code, Cpu, Gauge, Zap } from 'lucide-react'
import { SectionContainer, CodeBlock, BenefitCallout } from '@rbee/ui/molecules'

export function FeaturesSection() {
  return (
    <SectionContainer
      title="Enterprise-Grade Features. Homelab Simplicity."
      description="Pick a lane—API, GPUs, Scheduler, or Real-time—and see exactly how rbee fits your stack."
      bgVariant="secondary"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-10 md:space-y-14">
        <Tabs defaultValue="api" className="w-full">
          <TabsList
            className="grid w-full grid-cols-2 lg:grid-cols-4 h-auto rounded-xl border bg-card/60 p-1 gap-1"
            aria-label="Feature categories"
          >
            <TabsTrigger
              value="api"
              className="flex flex-col sm:flex-row items-center gap-2 py-3 data-[state=active]:bg-background data-[state=active]:shadow-sm data-[state=active]:text-foreground rounded-lg px-3 text-sm font-medium transition-colors"
            >
              <Code className="h-4 w-4" aria-hidden="true" />
              <span className="hidden sm:inline">OpenAI-Compatible</span>
              <span className="text-xs text-muted-foreground block leading-none sm:hidden">OpenAI</span>
            </TabsTrigger>
            <TabsTrigger
              value="gpu"
              className="flex flex-col sm:flex-row items-center gap-2 py-3 data-[state=active]:bg-background data-[state=active]:shadow-sm data-[state=active]:text-foreground rounded-lg px-3 text-sm font-medium transition-colors"
            >
              <Cpu className="h-4 w-4" aria-hidden="true" />
              <span className="hidden sm:inline">Multi-GPU</span>
              <span className="text-xs text-muted-foreground block leading-none sm:hidden">GPU</span>
            </TabsTrigger>
            <TabsTrigger
              value="scheduler"
              className="flex flex-col sm:flex-row items-center gap-2 py-3 data-[state=active]:bg-background data-[state=active]:shadow-sm data-[state=active]:text-foreground rounded-lg px-3 text-sm font-medium transition-colors"
            >
              <Gauge className="h-4 w-4" aria-hidden="true" />
              <span className="hidden sm:inline">Scheduler</span>
              <span className="text-xs text-muted-foreground block leading-none sm:hidden">Rhai</span>
            </TabsTrigger>
            <TabsTrigger
              value="sse"
              className="flex flex-col sm:flex-row items-center gap-2 py-3 data-[state=active]:bg-background data-[state=active]:shadow-sm data-[state=active]:text-foreground rounded-lg px-3 text-sm font-medium transition-colors"
            >
              <Zap className="h-4 w-4" aria-hidden="true" />
              <span className="hidden sm:inline">Real‑time</span>
              <span className="text-xs text-muted-foreground block leading-none sm:hidden">SSE</span>
            </TabsTrigger>
          </TabsList>

          <div aria-live="polite">
            <TabsContent value="api" className="mt-8">
              <div className="rounded-2xl border bg-card p-6 md:p-8 space-y-6 shadow-sm animate-in fade-in slide-in-from-bottom-2 duration-300 motion-reduce:animate-none">
                <div className="space-y-2">
                  <h3 className="text-2xl md:text-3xl font-semibold tracking-tight text-card-foreground">
                    OpenAI-Compatible API
                  </h3>
                  <p className="text-xs text-muted-foreground">Drop-in replacement for your existing tools</p>
                </div>

                <p className="text-base md:text-lg text-muted-foreground">
                  Drop-in for Zed, Cursor, Continue, or any OpenAI client. Keep your SDKs and prompts—just change the
                  base URL.
                </p>

                <div className="flex flex-wrap gap-2">
                  <span className="text-[11px] font-medium rounded-full bg-accent/60 text-foreground/90 px-2.5 py-1">
                    No API fees
                  </span>
                  <span className="text-[11px] font-medium rounded-full bg-accent/60 text-foreground/90 px-2.5 py-1">
                    Local tokens
                  </span>
                  <span className="text-[11px] font-medium rounded-full bg-accent/60 text-foreground/90 px-2.5 py-1">
                    Secure by default
                  </span>
                </div>

                <CodeBlock
                  title="Point clients to rbee"
                  language="bash"
                  copyable={true}
                  code={`# Before: Using OpenAI
export OPENAI_API_KEY=sk-...

# After: Using rbee (same code!)
export OPENAI_API_BASE=http://localhost:8080/v1

echo "→ Clients now talk to rbee at http://localhost:8080/v1"`}
                />

                <BenefitCallout variant="success" text="No code changes. Just point to localhost." />
              </div>
            </TabsContent>

            <TabsContent value="gpu" className="mt-8">
              <div className="rounded-2xl border bg-card p-6 md:p-8 space-y-6 shadow-sm animate-in fade-in slide-in-from-bottom-2 duration-300 motion-reduce:animate-none">
                <div className="space-y-2">
                  <h3 className="text-2xl md:text-3xl font-semibold tracking-tight text-card-foreground">
                    Multi-GPU Orchestration
                  </h3>
                  <p className="text-xs text-muted-foreground">Unified pool across all your hardware</p>
                </div>

                <p className="text-base md:text-lg text-muted-foreground">
                  Pool CUDA, Metal, and CPU backends. Mixed nodes act as one.
                </p>

                <div className="flex flex-wrap gap-2">
                  <span className="text-[11px] font-medium rounded-full bg-accent/60 text-foreground/90 px-2.5 py-1">
                    Multi-node
                  </span>
                  <span className="text-[11px] font-medium rounded-full bg-accent/60 text-foreground/90 px-2.5 py-1">
                    Backend-aware
                  </span>
                  <span className="text-[11px] font-medium rounded-full bg-accent/60 text-foreground/90 px-2.5 py-1">
                    Auto discovery
                  </span>
                </div>

                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                    <div className="flex-shrink-0 w-32 text-sm text-muted-foreground">RTX 4090 #1</div>
                    <div className="flex-1 h-8 bg-muted rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary flex items-center justify-end pr-2 transition-[width] duration-700 ease-out motion-reduce:transition-none"
                        style={{ width: '92%' }}
                      >
                        <span className="text-xs text-primary-foreground font-medium">92%</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="flex-shrink-0 w-32 text-sm text-muted-foreground">RTX 4090 #2</div>
                    <div className="flex-1 h-8 bg-muted rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary flex items-center justify-end pr-2 transition-[width] duration-700 ease-out motion-reduce:transition-none"
                        style={{ width: '88%' }}
                      >
                        <span className="text-xs text-primary-foreground font-medium">88%</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="flex-shrink-0 w-32 text-sm text-muted-foreground">M2 Ultra</div>
                    <div className="flex-1 h-8 bg-muted rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary flex items-center justify-end pr-2 transition-[width] duration-700 ease-out motion-reduce:transition-none"
                        style={{ width: '76%' }}
                      >
                        <span className="text-xs text-primary-foreground font-medium">76%</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="flex-shrink-0 w-32 text-sm text-muted-foreground">CPU Backend</div>
                    <div className="flex-1 h-8 bg-muted rounded-full overflow-hidden">
                      <div
                        className="h-full bg-chart-2 flex items-center justify-end pr-2 transition-[width] duration-700 ease-out motion-reduce:transition-none"
                        style={{ width: '34%' }}
                      >
                        <span className="text-xs text-primary-foreground font-medium">34%</span>
                      </div>
                    </div>
                  </div>
                </div>

                <p className="text-xs text-muted-foreground">
                  Live utilization varies per task; numbers here are illustrative.
                </p>

                <BenefitCallout variant="info" text="10× throughput by using all your hardware." />
              </div>
            </TabsContent>

            <TabsContent value="scheduler" className="mt-8">
              <div className="rounded-2xl border bg-card p-6 md:p-8 space-y-6 shadow-sm animate-in fade-in slide-in-from-bottom-2 duration-300 motion-reduce:animate-none">
                <div className="space-y-2">
                  <h3 className="text-2xl md:text-3xl font-semibold tracking-tight text-card-foreground">
                    Programmable Rhai Scheduler
                  </h3>
                  <p className="text-xs text-muted-foreground">Custom routing logic for your workloads</p>
                </div>

                <p className="text-base md:text-lg text-muted-foreground">
                  Route by model size, task type, labels, or compliance rules—your policy, your trade-offs.
                </p>

                <div className="flex flex-wrap gap-2">
                  <span className="text-[11px] font-medium rounded-full bg-accent/60 text-foreground/90 px-2.5 py-1">
                    Latency-aware
                  </span>
                  <span className="text-[11px] font-medium rounded-full bg-accent/60 text-foreground/90 px-2.5 py-1">
                    Cost caps
                  </span>
                  <span className="text-[11px] font-medium rounded-full bg-accent/60 text-foreground/90 px-2.5 py-1">
                    Compliance routes
                  </span>
                </div>

                <CodeBlock
                  title="Policy example"
                  language="rust"
                  copyable={true}
                  code={`// Route by model size, type, and labels
if task.model.ends_with("70b") { route_to("multi-gpu:labels=nvlink") }
else if task.kind == "image" { route_to("cuda:labels=rtx") }
else if task.region == "eu" { route_to("metal:labels=mac") }
else { route_to("cheapest") }`}
                />

                <BenefitCallout variant="primary" text="Optimize for cost, latency, or compliance—your rules." />
              </div>
            </TabsContent>

            <TabsContent value="sse" className="mt-8">
              <div className="rounded-2xl border bg-card p-6 md:p-8 space-y-6 shadow-sm animate-in fade-in slide-in-from-bottom-2 duration-300 motion-reduce:animate-none">
                <div className="space-y-2">
                  <h3 className="text-2xl md:text-3xl font-semibold tracking-tight text-card-foreground">
                    Task-Based API with SSE
                  </h3>
                  <p className="text-xs text-muted-foreground">Stream job lifecycle into your UI</p>
                </div>

                <p className="text-base md:text-lg text-muted-foreground">
                  Stream job lifecycle events—model loads, token output, cost—right into your UI.
                </p>

                <div className="flex flex-wrap gap-2">
                  <span className="text-[11px] font-medium rounded-full bg-accent/60 text-foreground/90 px-2.5 py-1">
                    Real-time
                  </span>
                  <span className="text-[11px] font-medium rounded-full bg-accent/60 text-foreground/90 px-2.5 py-1">
                    Back-pressure safe
                  </span>
                  <span className="text-[11px] font-medium rounded-full bg-accent/60 text-foreground/90 px-2.5 py-1">
                    Cost visible
                  </span>
                </div>

                <CodeBlock
                  title="SSE event stream"
                  language="json"
                  copyable={true}
                  code={`→ event: task.created
{ "id": "task_123", "status": "pending" }

→ event: model.loading
{ "progress": 0.45, "eta": "2.1s" }

→ event: token.generated
{ "token": "const", "total": 1 }

→ event: token.generated
{ "token": " api", "total": 2 }

→ event: task.completed
{ "id": "task_123", "status": "success", "total_tokens": 1234, "cost": "€0.00" }`}
                />

                <BenefitCallout text="Full visibility into every inference job." />
              </div>
            </TabsContent>
          </div>
        </Tabs>
      </div>
    </SectionContainer>
  )
}
