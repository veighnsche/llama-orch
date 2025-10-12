import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/atoms/Tabs/Tabs'
import { Code, Cpu, Gauge, Zap } from "lucide-react"

export function CoreFeaturesTabs() {
  return (
    <section className="py-24 bg-secondary">
      <div className="container mx-auto px-4">
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
                  <h3 className="text-2xl font-bold text-foreground mb-3">OpenAI-Compatible API</h3>
                  <p className="text-muted-foreground leading-relaxed">
                    Drop-in replacement for OpenAI API. Works with Zed, Cursor, Continue, and any tool that supports
                    OpenAI. No code changes required—just point to localhost.
                  </p>
                </div>

                <div className="bg-background rounded-lg p-6 font-mono text-sm">
                  <div className="text-muted-foreground"># Before: Using OpenAI</div>
                  <div className="text-chart-3 mt-2">export OPENAI_API_KEY=sk-...</div>
                  <div className="text-muted-foreground mt-4"># After: Using rbee (same code!)</div>
                  <div className="text-chart-3 mt-2">export OPENAI_API_BASE=http://localhost:8080/v1</div>
                </div>

                <div className="bg-chart-3/10 border border-chart-3/20 rounded-lg p-4">
                  <p className="text-chart-3 font-medium">✓ No code changes. Just point to localhost.</p>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="gpu" className="mt-8">
              <div className="bg-card border border-border rounded-lg p-8 space-y-6">
                <div>
                  <h3 className="text-2xl font-bold text-foreground mb-3">Multi-GPU Orchestration</h3>
                  <p className="text-muted-foreground leading-relaxed">
                    Orchestrate across the CUDA, Metal, and CPU backends you configure. Use every GPU you own across
                    your entire network.
                  </p>
                </div>

                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                    <div className="flex-shrink-0 w-32 text-sm text-muted-foreground">RTX 4090 #1</div>
                    <div className="flex-1 h-8 bg-secondary rounded-full overflow-hidden">
                      <div className="h-full bg-primary flex items-center justify-end pr-2" style={{ width: "92%" }}>
                        <span className="text-xs text-primary-foreground font-medium">92%</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="flex-shrink-0 w-32 text-sm text-muted-foreground">RTX 4090 #2</div>
                    <div className="flex-1 h-8 bg-secondary rounded-full overflow-hidden">
                      <div className="h-full bg-primary flex items-center justify-end pr-2" style={{ width: "88%" }}>
                        <span className="text-xs text-primary-foreground font-medium">88%</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="flex-shrink-0 w-32 text-sm text-muted-foreground">M2 Ultra</div>
                    <div className="flex-1 h-8 bg-secondary rounded-full overflow-hidden">
                      <div className="h-full bg-primary flex items-center justify-end pr-2" style={{ width: "76%" }}>
                        <span className="text-xs text-primary-foreground font-medium">76%</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="flex-shrink-0 w-32 text-sm text-muted-foreground">CPU Backend</div>
                    <div className="flex-1 h-8 bg-secondary rounded-full overflow-hidden">
                      <div className="h-full bg-chart-2 flex items-center justify-end pr-2" style={{ width: "34%" }}>
                        <span className="text-xs text-primary-foreground font-medium">34%</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-chart-2/10 border border-chart-2/20 rounded-lg p-4">
                  <p className="text-chart-2 font-medium">✓ 10x throughput by using all your hardware.</p>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="scheduler" className="mt-8">
              <div className="bg-card border border-border rounded-lg p-8 space-y-6">
                <div>
                  <h3 className="text-2xl font-bold text-foreground mb-3">Programmable Rhai Scheduler</h3>
                  <p className="text-muted-foreground leading-relaxed">
                    Write custom routing logic in Rhai. Route large models to multi-GPU setups, image generation to
                    CUDA, everything else to cheapest.
                  </p>
                </div>

                <div className="bg-background rounded-lg p-6 font-mono text-sm">
                  <div className="text-muted-foreground">// Custom routing logic</div>
                  <div className="text-chart-4 mt-2">if</div>
                  <div className="text-foreground">{' task.model.contains("70b") {'}</div>
                  <div className="text-foreground pl-4">
                    route_to(<span className="text-chart-3">"multi-gpu-cluster"</span>)
                  </div>
                  <div className="text-foreground">{"}"}</div>
                  <div className="text-chart-4 mt-2">else if</div>
                  <div className="text-foreground">{' task.type == "image" {'}</div>
                  <div className="text-foreground pl-4">
                    route_to(<span className="text-chart-3">"cuda-only"</span>)
                  </div>
                  <div className="text-foreground">{"}"}</div>
                  <div className="text-chart-4 mt-2">else</div>
                  <div className="text-foreground">{" {"}</div>
                  <div className="text-foreground pl-4">
                    route_to(<span className="text-chart-3">"cheapest"</span>)
                  </div>
                  <div className="text-foreground">{"}"}</div>
                </div>

                <div className="bg-primary/10 border border-primary/20 rounded-lg p-4">
                  <p className="text-primary font-medium">✓ Optimize for cost, latency, or compliance—your rules.</p>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="sse" className="mt-8">
              <div className="bg-card border border-border rounded-lg p-8 space-y-6">
                <div>
                  <h3 className="text-2xl font-bold text-foreground mb-3">Task-Based API with SSE</h3>
                  <p className="text-muted-foreground leading-relaxed">
                    Real‑time progress updates. See model loading, token generation, and cost tracking as it happens.
                  </p>
                </div>

                <div className="bg-background rounded-lg p-6 font-mono text-sm space-y-2">
                  <div className="text-muted-foreground">→ event: task.created</div>
                  <div className="text-foreground pl-4">{'{ "id": "task_123", "status": "pending" }'}</div>
                  <div className="text-muted-foreground mt-2">→ event: model.loading</div>
                  <div className="text-foreground pl-4">{'{ "progress": 0.45, "eta": "2.1s" }'}</div>
                  <div className="text-muted-foreground mt-2">→ event: token.generated</div>
                  <div className="text-foreground pl-4">{'{ "token": "const", "total": 1 }'}</div>
                  <div className="text-muted-foreground mt-2">→ event: token.generated</div>
                  <div className="text-foreground pl-4">{'{ "token": " api", "total": 2 }'}</div>
                </div>

                <div className="bg-secondary border border-border rounded-lg p-4">
                  <p className="text-foreground font-medium">✓ Full visibility into every inference job.</p>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </section>
  )
}
