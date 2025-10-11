import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Code, Cpu, Gauge, Zap } from "lucide-react"

export function FeaturesSection() {
  return (
    <section className="py-24 bg-slate-50">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-slate-900 mb-6 text-balance">
            Enterprise-Grade Features. Homelab Simplicity.
          </h2>
        </div>

        <div className="max-w-5xl mx-auto">
          <Tabs defaultValue="api" className="w-full">
            <TabsList className="grid w-full grid-cols-2 lg:grid-cols-4 h-auto">
              <TabsTrigger value="api" className="flex items-center gap-2 py-3">
                <Code className="h-4 w-4" />
                <span className="hidden sm:inline">OpenAI API</span>
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
                <span className="hidden sm:inline">Real-time</span>
                <span className="sm:hidden">SSE</span>
              </TabsTrigger>
            </TabsList>

            <TabsContent value="api" className="mt-8">
              <div className="bg-white border border-slate-200 rounded-lg p-8 space-y-6">
                <div>
                  <h3 className="text-2xl font-bold text-slate-900 mb-3">OpenAI-Compatible API</h3>
                  <p className="text-slate-600 leading-relaxed">
                    Drop-in replacement for OpenAI API. Works with Zed, Cursor, Continue, and any tool that supports
                    OpenAI.
                  </p>
                </div>

                <div className="bg-slate-900 rounded-lg p-6 font-mono text-sm">
                  <div className="text-slate-400"># Before: Using OpenAI</div>
                  <div className="text-green-400 mt-2">export OPENAI_API_KEY=sk-...</div>
                  <div className="text-slate-400 mt-4"># After: Using rbee (same code!)</div>
                  <div className="text-green-400 mt-2">export OPENAI_API_BASE=http://localhost:8080/v1</div>
                </div>

                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <p className="text-green-900 font-medium">✓ No code changes. Just point to localhost.</p>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="gpu" className="mt-8">
              <div className="bg-white border border-slate-200 rounded-lg p-8 space-y-6">
                <div>
                  <h3 className="text-2xl font-bold text-slate-900 mb-3">Multi-GPU Orchestration</h3>
                  <p className="text-slate-600 leading-relaxed">
                    Automatically distribute workloads across CUDA, Metal, and CPU backends. Use every GPU you own.
                  </p>
                </div>

                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                    <div className="flex-shrink-0 w-32 text-sm text-slate-600">RTX 4090 #1</div>
                    <div className="flex-1 h-8 bg-slate-100 rounded-full overflow-hidden">
                      <div className="h-full bg-amber-500 flex items-center justify-end pr-2" style={{ width: "92%" }}>
                        <span className="text-xs text-white font-medium">92%</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="flex-shrink-0 w-32 text-sm text-slate-600">RTX 4090 #2</div>
                    <div className="flex-1 h-8 bg-slate-100 rounded-full overflow-hidden">
                      <div className="h-full bg-amber-500 flex items-center justify-end pr-2" style={{ width: "88%" }}>
                        <span className="text-xs text-white font-medium">88%</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="flex-shrink-0 w-32 text-sm text-slate-600">M2 Ultra</div>
                    <div className="flex-1 h-8 bg-slate-100 rounded-full overflow-hidden">
                      <div className="h-full bg-amber-500 flex items-center justify-end pr-2" style={{ width: "76%" }}>
                        <span className="text-xs text-white font-medium">76%</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="flex-shrink-0 w-32 text-sm text-slate-600">CPU Fallback</div>
                    <div className="flex-1 h-8 bg-slate-100 rounded-full overflow-hidden">
                      <div className="h-full bg-blue-400 flex items-center justify-end pr-2" style={{ width: "34%" }}>
                        <span className="text-xs text-white font-medium">34%</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <p className="text-blue-900 font-medium">✓ 10x throughput by using all your hardware.</p>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="scheduler" className="mt-8">
              <div className="bg-white border border-slate-200 rounded-lg p-8 space-y-6">
                <div>
                  <h3 className="text-2xl font-bold text-slate-900 mb-3">Programmable Rhai Scheduler</h3>
                  <p className="text-slate-600 leading-relaxed">
                    Write custom routing logic in Rhai. Route large models to multi-GPU setups, image generation to
                    CUDA, everything else to cheapest.
                  </p>
                </div>

                <div className="bg-slate-900 rounded-lg p-6 font-mono text-sm">
                  <div className="text-slate-400">// Custom routing logic</div>
                  <div className="text-purple-400 mt-2">if</div>
                  <div className="text-slate-300">{' task.model.contains("70b") {'}</div>
                  <div className="text-slate-300 pl-4">
                    route_to(<span className="text-green-300">"multi-gpu-cluster"</span>)
                  </div>
                  <div className="text-slate-300">{"}"}</div>
                  <div className="text-purple-400 mt-2">else if</div>
                  <div className="text-slate-300">{' task.type == "image" {'}</div>
                  <div className="text-slate-300 pl-4">
                    route_to(<span className="text-green-300">"cuda-only"</span>)
                  </div>
                  <div className="text-slate-300">{"}"}</div>
                  <div className="text-purple-400 mt-2">else</div>
                  <div className="text-slate-300">{" {"}</div>
                  <div className="text-slate-300 pl-4">
                    route_to(<span className="text-green-300">"cheapest"</span>)
                  </div>
                  <div className="text-slate-300">{"}"}</div>
                </div>

                <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
                  <p className="text-amber-900 font-medium">✓ Optimize for cost, latency, or compliance—your rules.</p>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="sse" className="mt-8">
              <div className="bg-white border border-slate-200 rounded-lg p-8 space-y-6">
                <div>
                  <h3 className="text-2xl font-bold text-slate-900 mb-3">Task-Based API with SSE</h3>
                  <p className="text-slate-600 leading-relaxed">
                    Real-time progress updates. See model loading, token generation, and cost tracking as it happens.
                  </p>
                </div>

                <div className="bg-slate-900 rounded-lg p-6 font-mono text-sm space-y-2">
                  <div className="text-slate-400">→ event: task.created</div>
                  <div className="text-slate-300 pl-4">{'{ "id": "task_123", "status": "pending" }'}</div>
                  <div className="text-slate-400 mt-2">→ event: model.loading</div>
                  <div className="text-slate-300 pl-4">{'{ "progress": 0.45, "eta": "2.1s" }'}</div>
                  <div className="text-slate-400 mt-2">→ event: token.generated</div>
                  <div className="text-slate-300 pl-4">{'{ "token": "const", "total": 1 }'}</div>
                  <div className="text-slate-400 mt-2">→ event: token.generated</div>
                  <div className="text-slate-300 pl-4">{'{ "token": " api", "total": 2 }'}</div>
                </div>

                <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
                  <p className="text-slate-900 font-medium">✓ Full visibility into every inference job.</p>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </section>
  )
}
