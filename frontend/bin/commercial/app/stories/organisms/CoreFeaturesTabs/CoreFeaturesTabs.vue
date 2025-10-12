<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-008: Implemented CoreFeaturesTabs -->
<script setup lang="ts">
import { Tabs, TabsList, TabsTrigger, TabsContent } from '~/stories'
import { Code, Cpu, Gauge, Zap } from 'lucide-vue-next'
</script>

<template>
  <section class="py-24 bg-secondary">
    <div class="container mx-auto px-4">
      <div class="max-w-5xl mx-auto">
        <Tabs default-value="api" class="w-full">
          <TabsList class="grid w-full grid-cols-2 lg:grid-cols-4 h-auto">
            <TabsTrigger value="api" class="flex items-center gap-2 py-3">
              <Code class="h-4 w-4" />
              <span class="hidden sm:inline">OpenAI API</span>
              <span class="sm:hidden">API</span>
            </TabsTrigger>
            <TabsTrigger value="gpu" class="flex items-center gap-2 py-3">
              <Cpu class="h-4 w-4" />
              <span class="hidden sm:inline">Multi-GPU</span>
              <span class="sm:hidden">GPU</span>
            </TabsTrigger>
            <TabsTrigger value="scheduler" class="flex items-center gap-2 py-3">
              <Gauge class="h-4 w-4" />
              <span class="hidden sm:inline">Scheduler</span>
              <span class="sm:hidden">Rhai</span>
            </TabsTrigger>
            <TabsTrigger value="sse" class="flex items-center gap-2 py-3">
              <Zap class="h-4 w-4" />
              <span class="hidden sm:inline">Real-time</span>
              <span class="sm:hidden">SSE</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="api" class="mt-8">
            <div class="bg-card border border-border rounded-lg p-8 space-y-6">
              <div>
                <h3 class="text-2xl font-bold text-foreground mb-3">OpenAI-Compatible API</h3>
                <p class="text-muted-foreground leading-relaxed">
                  Drop-in replacement for OpenAI API. Works with Zed, Cursor, Continue, and any tool that supports
                  OpenAI. No code changes required—just point to localhost.
                </p>
              </div>

              <div class="bg-secondary rounded-lg p-6 font-mono text-sm">
                <div class="text-muted-foreground"># Before: Using OpenAI</div>
                <div class="text-accent mt-2">export OPENAI_API_KEY=sk-...</div>
                <div class="text-muted-foreground mt-4"># After: Using rbee (same code!)</div>
                <div class="text-accent mt-2">export OPENAI_API_BASE=http://localhost:8080/v1</div>
              </div>

              <div class="bg-accent/10 border border-accent/30 rounded-lg p-4">
                <p class="text-accent-foreground font-medium">✓ No code changes. Just point to localhost.</p>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="gpu" class="mt-8">
            <div class="bg-card border border-border rounded-lg p-8 space-y-6">
              <div>
                <h3 class="text-2xl font-bold text-foreground mb-3">Multi-GPU Orchestration</h3>
                <p class="text-muted-foreground leading-relaxed">
                  Automatically distribute workloads across CUDA, Metal, and CPU backends. Use every GPU you own
                  across your entire network.
                </p>
              </div>

              <div class="space-y-3">
                <div class="flex items-center gap-3">
                  <div class="flex-shrink-0 w-32 text-sm text-muted-foreground">RTX 4090 #1</div>
                  <div class="flex-1 h-8 bg-muted rounded-full overflow-hidden">
                    <div class="h-full bg-primary flex items-center justify-end pr-2" style="width: 92%">
                      <span class="text-xs text-primary-foreground font-medium">92%</span>
                    </div>
                  </div>
                </div>
                <div class="flex items-center gap-3">
                  <div class="flex-shrink-0 w-32 text-sm text-muted-foreground">RTX 4090 #2</div>
                  <div class="flex-1 h-8 bg-muted rounded-full overflow-hidden">
                    <div class="h-full bg-primary flex items-center justify-end pr-2" style="width: 88%">
                      <span class="text-xs text-primary-foreground font-medium">88%</span>
                    </div>
                  </div>
                </div>
                <div class="flex items-center gap-3">
                  <div class="flex-shrink-0 w-32 text-sm text-muted-foreground">M2 Ultra</div>
                  <div class="flex-1 h-8 bg-muted rounded-full overflow-hidden">
                    <div class="h-full bg-primary flex items-center justify-end pr-2" style="width: 76%">
                      <span class="text-xs text-primary-foreground font-medium">76%</span>
                    </div>
                  </div>
                </div>
                <div class="flex items-center gap-3">
                  <div class="flex-shrink-0 w-32 text-sm text-muted-foreground">CPU Fallback</div>
                  <div class="flex-1 h-8 bg-muted rounded-full overflow-hidden">
                    <div class="h-full bg-blue-400 flex items-center justify-end pr-2" style="width: 34%">
                      <span class="text-xs text-white font-medium">34%</span>
                    </div>
                  </div>
                </div>
              </div>

              <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <p class="text-blue-900 font-medium">✓ 10x throughput by using all your hardware.</p>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="scheduler" class="mt-8">
            <div class="bg-card border border-border rounded-lg p-8 space-y-6">
              <div>
                <h3 class="text-2xl font-bold text-foreground mb-3">Programmable Rhai Scheduler</h3>
                <p class="text-muted-foreground leading-relaxed">
                  Write custom routing logic in Rhai. Route large models to multi-GPU setups, image generation to
                  CUDA, everything else to cheapest.
                </p>
              </div>

              <div class="bg-secondary rounded-lg p-6 font-mono text-sm">
                <div class="text-muted-foreground">// Custom routing logic</div>
                <div class="text-purple-400 mt-2">if</div>
                <div class="text-foreground"> task.model.contains("70b") {</div>
                <div class="text-foreground pl-4">
                  route_to(<span class="text-accent">"multi-gpu-cluster"</span>)
                </div>
                <div class="text-foreground">}</div>
                <div class="text-purple-400 mt-2">else if</div>
                <div class="text-foreground"> task.type == "image" {</div>
                <div class="text-foreground pl-4">
                  route_to(<span class="text-accent">"cuda-only"</span>)
                </div>
                <div class="text-foreground">}</div>
                <div class="text-purple-400 mt-2">else</div>
                <div class="text-foreground"> {</div>
                <div class="text-foreground pl-4">
                  route_to(<span class="text-accent">"cheapest"</span>)
                </div>
                <div class="text-foreground">}</div>
              </div>

              <div class="bg-primary/10 border border-primary/30 rounded-lg p-4">
                <p class="text-primary font-medium">✓ Optimize for cost, latency, or compliance—your rules.</p>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="sse" class="mt-8">
            <div class="bg-card border border-border rounded-lg p-8 space-y-6">
              <div>
                <h3 class="text-2xl font-bold text-foreground mb-3">Task-Based API with SSE</h3>
                <p class="text-muted-foreground leading-relaxed">
                  Real-time progress updates. See model loading, token generation, and cost tracking as it happens.
                </p>
              </div>

              <div class="bg-secondary rounded-lg p-6 font-mono text-sm space-y-2">
                <div class="text-muted-foreground">→ event: task.created</div>
                <div class="text-foreground pl-4">{ "id": "task_123", "status": "pending" }</div>
                <div class="text-muted-foreground mt-2">→ event: model.loading</div>
                <div class="text-foreground pl-4">{ "progress": 0.45, "eta": "2.1s" }</div>
                <div class="text-muted-foreground mt-2">→ event: token.generated</div>
                <div class="text-foreground pl-4">{ "token": "const", "total": 1 }</div>
                <div class="text-muted-foreground mt-2">→ event: token.generated</div>
                <div class="text-foreground pl-4">{ "token": " api", "total": 2 }</div>
              </div>

              <div class="bg-muted border border-border rounded-lg p-4">
                <p class="text-foreground font-medium">✓ Full visibility into every inference job.</p>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  </section>
</template>
