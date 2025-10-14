import { Tabs, TabsContent, TabsList, TabsTrigger } from '@rbee/ui/atoms/Tabs'
import { Badge } from '@rbee/ui/atoms/Badge'
import { Code, Cpu, Gauge, Zap, Check } from 'lucide-react'

export function CoreFeaturesTabs() {
  return (
    <section id="feature-list" className="py-24 bg-gradient-to-b from-secondary to-background">
      <div className="container mx-auto px-4">
        <Tabs defaultValue="api" className="w-full" orientation="horizontal">
          <div className="grid gap-8 lg:grid-cols-[320px_minmax(0,1fr)]">
            {/* Left rail: sticky intro + TabsList */}
            <div className="lg:sticky lg:top-24 self-start space-y-6">
              <div>
                <h2 className="text-3xl font-bold tracking-tight text-foreground">Core capabilities</h2>
                <p className="mt-2 text-muted-foreground">
                  Swap in the API, scale across your hardware, route with code, and watch jobs stream in real time.
                </p>
              </div>

              <TabsList aria-label="Core features">
                <TabsTrigger value="api" className="flex-col lg:flex-row items-start lg:items-center">
                  <span className="flex items-center gap-3 w-full">
                    <Code className="size-4 text-muted-foreground group-data-[state=active]:text-primary" />
                    <span className="font-semibold">
                      <span className="hidden sm:inline">OpenAI-Compatible</span>
                      <span className="sm:hidden">API</span>
                    </span>
                  </span>
                  <span className="mt-0.5 block text-xs text-muted-foreground hidden lg:block w-full">Drop-in API</span>
                </TabsTrigger>

                <TabsTrigger value="gpu" className="flex-col lg:flex-row items-start lg:items-center">
                  <span className="flex items-center gap-3 w-full">
                    <Cpu className="size-4 text-muted-foreground group-data-[state=active]:text-primary" />
                    <span className="font-semibold">
                      <span className="hidden sm:inline">Multi-GPU</span>
                      <span className="sm:hidden">GPU</span>
                    </span>
                  </span>
                  <span className="mt-0.5 block text-xs text-muted-foreground hidden lg:block w-full">Use every GPU</span>
                </TabsTrigger>

                <TabsTrigger value="scheduler" className="flex-col lg:flex-row items-start lg:items-center">
                  <span className="flex items-center gap-3 w-full">
                    <Gauge className="size-4 text-muted-foreground group-data-[state=active]:text-primary" />
                    <span className="font-semibold">
                      <span className="hidden sm:inline">Scheduler</span>
                      <span className="sm:hidden">Rhai</span>
                    </span>
                  </span>
                  <span className="mt-0.5 block text-xs text-muted-foreground hidden lg:block w-full">Route with Rhai</span>
                </TabsTrigger>

                <TabsTrigger value="sse" className="flex-col lg:flex-row items-start lg:items-center">
                  <span className="flex items-center gap-3 w-full">
                    <Zap className="size-4 text-muted-foreground group-data-[state=active]:text-primary" />
                    <span className="font-semibold">
                      <span className="hidden sm:inline">Real‑time</span>
                      <span className="sm:hidden">SSE</span>
                    </span>
                  </span>
                  <span className="mt-0.5 block text-xs text-muted-foreground hidden lg:block w-full">Live job stream</span>
                </TabsTrigger>
              </TabsList>
            </div>

            {/* Right column: content panels */}
            <div className="relative min-h-[600px]">
              <TabsContent value="api" className="animate-in fade-in slide-in-from-right-4 duration-300 data-[state=inactive]:absolute data-[state=inactive]:invisible">
                <div className="bg-card border border-border rounded-2xl p-8 space-y-6">
                  <div>
                    <div className="flex items-center gap-2 mb-3">
                      <h3 className="text-2xl font-bold text-foreground">OpenAI-compatible API</h3>
                      <Badge variant="secondary">Drop-in</Badge>
                    </div>
                    <p className="text-muted-foreground leading-relaxed">
                      Swap endpoints, keep your code. Works with Zed, Cursor, Continue—any OpenAI client.
                    </p>
                  </div>

                  <div className="bg-background rounded-lg p-6 font-mono text-sm">
                    <div className="text-muted-foreground"># Before: OpenAI</div>
                    <div className="text-chart-3 mt-2">export OPENAI_API_KEY=sk-...</div>
                    <div className="text-muted-foreground mt-4"># After: rbee (same code)</div>
                    <div className="text-chart-3 mt-2">export OPENAI_API_BASE=http://localhost:8080/v1</div>
                  </div>

                  <div className="bg-chart-3/10 border border-chart-3/20 rounded-lg p-4 flex items-center gap-2">
                    <Check className="size-4 text-chart-3 shrink-0" />
                    <p className="text-chart-3 font-medium">Drop-in replacement. Point to localhost.</p>
                  </div>

                  <div className="pt-2">
                    <h4 className="text-sm font-semibold text-foreground mb-3">Why it matters</h4>
                    <ul className="grid sm:grid-cols-3 gap-3 text-sm text-muted-foreground">
                      <li className="flex items-center gap-2">
                        <Check className="size-4 text-primary shrink-0" />
                        <span>No vendor lock-in</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <Check className="size-4 text-primary shrink-0" />
                        <span>Use your models + GPUs</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <Check className="size-4 text-primary shrink-0" />
                        <span>Keep existing tooling</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="gpu" className="animate-in fade-in slide-in-from-right-4 duration-300 data-[state=inactive]:absolute data-[state=inactive]:invisible">
                <div className="bg-card border border-border rounded-2xl p-8 space-y-6">
                  <div>
                    <div className="flex items-center gap-2 mb-3">
                      <h3 className="text-2xl font-bold text-foreground">Multi-GPU orchestration</h3>
                      <Badge variant="secondary">Scale</Badge>
                    </div>
                    <p className="text-muted-foreground leading-relaxed">
                      Run across CUDA, Metal, and CPU backends. Use every GPU across your network.
                    </p>
                  </div>

                  <div className="space-y-3">
                    <div className="flex items-center gap-3">
                      <div className="w-36 text-xs text-muted-foreground">RTX 4090 #1</div>
                      <div className="flex-1 h-3 bg-muted rounded-full overflow-hidden">
                        <div
                          className="h-full bg-primary animate-in grow-in origin-left"
                          style={{ width: '92%' }}
                          role="progressbar"
                          aria-valuenow={92}
                          aria-valuemin={0}
                          aria-valuemax={100}
                          aria-label="RTX 4090 #1 utilization"
                        />
                      </div>
                      <span className="text-xs text-muted-foreground w-10 text-right">92%</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="w-36 text-xs text-muted-foreground">RTX 4090 #2</div>
                      <div className="flex-1 h-3 bg-muted rounded-full overflow-hidden">
                        <div
                          className="h-full bg-primary animate-in grow-in origin-left"
                          style={{ width: '88%' }}
                          role="progressbar"
                          aria-valuenow={88}
                          aria-valuemin={0}
                          aria-valuemax={100}
                          aria-label="RTX 4090 #2 utilization"
                        />
                      </div>
                      <span className="text-xs text-muted-foreground w-10 text-right">88%</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="w-36 text-xs text-muted-foreground">M2 Ultra</div>
                      <div className="flex-1 h-3 bg-muted rounded-full overflow-hidden">
                        <div
                          className="h-full bg-primary animate-in grow-in origin-left"
                          style={{ width: '76%' }}
                          role="progressbar"
                          aria-valuenow={76}
                          aria-valuemin={0}
                          aria-valuemax={100}
                          aria-label="M2 Ultra utilization"
                        />
                      </div>
                      <span className="text-xs text-muted-foreground w-10 text-right">76%</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="w-36 text-xs text-muted-foreground">CPU Backend</div>
                      <div className="flex-1 h-3 bg-muted rounded-full overflow-hidden">
                        <div
                          className="h-full bg-chart-2 animate-in grow-in origin-left"
                          style={{ width: '34%' }}
                          role="progressbar"
                          aria-valuenow={34}
                          aria-valuemin={0}
                          aria-valuemax={100}
                          aria-label="CPU Backend utilization"
                        />
                      </div>
                      <span className="text-xs text-muted-foreground w-10 text-right">34%</span>
                    </div>
                  </div>

                  <div className="bg-chart-2/10 border border-chart-2/20 rounded-lg p-4 flex items-center gap-2">
                    <Check className="size-4 text-chart-2 shrink-0" />
                    <p className="text-chart-2 font-medium">Higher throughput by saturating all devices.</p>
                  </div>

                  <div className="pt-2">
                    <h4 className="text-sm font-semibold text-foreground mb-3">Why it matters</h4>
                    <ul className="grid sm:grid-cols-3 gap-3 text-sm text-muted-foreground">
                      <li className="flex items-center gap-2">
                        <Check className="size-4 text-primary shrink-0" />
                        <span>Bigger models fit</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <Check className="size-4 text-primary shrink-0" />
                        <span>Lower latency under load</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <Check className="size-4 text-primary shrink-0" />
                        <span>No single-machine bottleneck</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="scheduler" className="animate-in fade-in slide-in-from-right-4 duration-300 data-[state=inactive]:absolute data-[state=inactive]:invisible">
                <div className="bg-card border border-border rounded-2xl p-8 space-y-6">
                  <div>
                    <div className="flex items-center gap-2 mb-3">
                      <h3 className="text-2xl font-bold text-foreground">Programmable scheduler (Rhai)</h3>
                      <Badge variant="secondary">Control</Badge>
                    </div>
                    <p className="text-muted-foreground leading-relaxed">
                      Write routing rules. Send 70B to multi-GPU, images to CUDA, everything else to cheapest.
                    </p>
                  </div>

                  <div className="bg-background rounded-lg p-6 font-mono text-sm">
                    <div className="text-muted-foreground">// Custom routing logic</div>
                    <div className="text-chart-4 mt-2">if</div>
                    <div className="text-foreground">{' task.model.contains("70b") {'}</div>
                    <div className="text-foreground pl-4">
                      route_to(<span className="text-chart-3">"multi-gpu-cluster"</span>)
                    </div>
                    <div className="text-foreground">{'}'}</div>
                    <div className="text-chart-4 mt-2">else if</div>
                    <div className="text-foreground">{' task.type == "image" {'}</div>
                    <div className="text-foreground pl-4">
                      route_to(<span className="text-chart-3">"cuda-only"</span>)
                    </div>
                    <div className="text-foreground">{'}'}</div>
                    <div className="text-chart-4 mt-2">else</div>
                    <div className="text-foreground">{' {'}</div>
                    <div className="text-foreground pl-4">
                      route_to(<span className="text-chart-3">"cheapest"</span>)
                    </div>
                    <div className="text-foreground">{'}'}</div>
                  </div>

                  <div className="bg-primary/10 border border-primary/20 rounded-lg p-4 flex items-center gap-2">
                    <Check className="size-4 text-primary shrink-0" />
                    <p className="text-primary font-medium">Optimize for cost, latency, or compliance—your rules.</p>
                  </div>

                  <div className="pt-2">
                    <h4 className="text-sm font-semibold text-foreground mb-3">Why it matters</h4>
                    <ul className="grid sm:grid-cols-3 gap-3 text-sm text-muted-foreground">
                      <li className="flex items-center gap-2">
                        <Check className="size-4 text-primary shrink-0" />
                        <span>Deterministic routing</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <Check className="size-4 text-primary shrink-0" />
                        <span>Policy & compliance ready</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <Check className="size-4 text-primary shrink-0" />
                        <span>Easy to evolve</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="sse" className="animate-in fade-in slide-in-from-right-4 duration-300 data-[state=inactive]:absolute data-[state=inactive]:invisible">
                <div className="bg-card border border-border rounded-2xl p-8 space-y-6">
                  <div>
                    <div className="flex items-center gap-2 mb-3">
                      <h3 className="text-2xl font-bold text-foreground">Task-based API with SSE</h3>
                      <Badge variant="secondary">Observe</Badge>
                    </div>
                    <p className="text-muted-foreground leading-relaxed">
                      See model loading, token generation, and costs stream in as they happen.
                    </p>
                  </div>

                  <div className="bg-background rounded-lg p-6 font-mono text-sm" role="log" aria-live="polite">
                    <div className="space-y-2">
                      <div role="status">
                        <div className="text-muted-foreground">→ event: task.created</div>
                        <div className="text-foreground pl-4">{'{ "id": "task_123", "status": "pending" }'}</div>
                      </div>
                      <div role="status">
                        <div className="text-muted-foreground mt-2">→ event: model.loading</div>
                        <div className="text-foreground pl-4">{'{ "progress": 0.45, "eta": "2.1s" }'}</div>
                      </div>
                      <div role="status">
                        <div className="text-muted-foreground mt-2">→ event: token.generated</div>
                        <div className="text-foreground pl-4">{'{ "token": "const", "total": 1 }'}</div>
                      </div>
                      <div role="status">
                        <div className="text-muted-foreground mt-2">→ event: token.generated</div>
                        <div className="text-foreground pl-4">{'{ "token": " api", "total": 2 }'}</div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-secondary border border-border rounded-lg p-4 flex items-center gap-2">
                    <Check className="size-4 text-foreground shrink-0" />
                    <p className="text-foreground font-medium">Full visibility for every inference job.</p>
                  </div>

                  <div className="pt-2">
                    <h4 className="text-sm font-semibold text-foreground mb-3">Why it matters</h4>
                    <ul className="grid sm:grid-cols-3 gap-3 text-sm text-muted-foreground">
                      <li className="flex items-center gap-2">
                        <Check className="size-4 text-primary shrink-0" />
                        <span>Faster debugging</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <Check className="size-4 text-primary shrink-0" />
                        <span>UX you can trust</span>
                      </li>
                      <li className="flex items-center gap-2">
                        <Check className="size-4 text-primary shrink-0" />
                        <span>Accurate cost tracking</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </TabsContent>
            </div>
          </div>
        </Tabs>
      </div>
    </section>
  )
}
