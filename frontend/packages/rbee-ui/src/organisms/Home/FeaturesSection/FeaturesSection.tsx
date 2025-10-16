'use client'

import { Alert, AlertDescription } from '@rbee/ui/atoms/Alert'
import { Tabs, TabsContent, TabsList } from '@rbee/ui/atoms/Tabs'
import {
  CodeBlock,
  FeatureBadge,
  FeatureHeader,
  FeatureTab,
  FeatureTabContent,
  GPUUtilizationBar,
  SectionContainer,
} from '@rbee/ui/molecules'
import { Code, Cpu, Gauge, Zap } from 'lucide-react'

export function FeaturesSection() {
  return (
    <SectionContainer
      title="Enterprise-Grade Features. Homelab Simplicity."
      description="Pick a lane—API, GPUs, Scheduler, or Real-time—and see exactly how rbee fits your stack."
      bgVariant="secondary"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-10 md:space-y-14">
        <Tabs defaultValue="api" className="w-full">
          <div className="rounded-xl border border-border bg-card/60 p-1">
            <TabsList
              className="grid w-full grid-cols-2 lg:grid-cols-4 h-auto gap-2 rounded-none border-0 bg-transparent p-0"
              aria-label="Feature categories"
            >
              <FeatureTab value="api" icon={<Code className="w-6 h-6" />} label="OpenAI-Compatible" mobileLabel="OpenAI" />
              <FeatureTab value="gpu" icon={<Cpu className="w-6 h-6" />} label="Multi-GPU" mobileLabel="GPU" />
              <FeatureTab value="scheduler" icon={<Gauge className="w-6 h-6" />} label="Scheduler" mobileLabel="Rhai" />
              <FeatureTab value="sse" icon={<Zap className="w-6 h-6" />} label="Real‑time" mobileLabel="SSE" />
            </TabsList>
          </div>

          <div aria-live="polite">
            <TabsContent value="api" className="mt-8">
              <FeatureTabContent>
                <FeatureHeader title="OpenAI-Compatible API" subtitle="Drop-in replacement for your existing tools" />

                <p className="text-base md:text-lg text-muted-foreground">
                  Drop-in for Zed, Cursor, Continue, or any OpenAI client. Keep your SDKs and prompts—just change the
                  base URL.
                </p>

                <div className="flex flex-wrap gap-2">
                  <FeatureBadge label="No API fees" />
                  <FeatureBadge label="Local tokens" />
                  <FeatureBadge label="Secure by default" />
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

                <Alert variant="success">
                  <AlertDescription>No code changes. Just point to localhost.</AlertDescription>
                </Alert>
              </FeatureTabContent>
            </TabsContent>

            <TabsContent value="gpu" className="mt-8">
              <FeatureTabContent>
                <FeatureHeader title="Multi-GPU Orchestration" subtitle="Unified pool across all your hardware" />

                <p className="text-base md:text-lg text-muted-foreground">
                  Pool CUDA, Metal, and CPU backends. Mixed nodes act as one.
                </p>

                <div className="flex flex-wrap gap-2">
                  <FeatureBadge label="Multi-node" />
                  <FeatureBadge label="Backend-aware" />
                  <FeatureBadge label="Auto discovery" />
                </div>

                <div className="space-y-3">
                  <GPUUtilizationBar label="RTX 4090 #1" percentage={92} />
                  <GPUUtilizationBar label="RTX 4090 #2" percentage={88} />
                  <GPUUtilizationBar label="M2 Ultra" percentage={76} />
                  <GPUUtilizationBar label="CPU Backend" percentage={34} variant="secondary" />
                </div>

                <p className="text-xs text-muted-foreground">
                  Live utilization varies per task; numbers here are illustrative.
                </p>

                <Alert variant="info">
                  <AlertDescription>10× throughput by using all your hardware.</AlertDescription>
                </Alert>
              </FeatureTabContent>
            </TabsContent>

            <TabsContent value="scheduler" className="mt-8">
              <FeatureTabContent>
                <FeatureHeader title="Programmable Rhai Scheduler" subtitle="Custom routing logic for your workloads" />

                <p className="text-base md:text-lg text-muted-foreground">
                  Route by model size, task type, labels, or compliance rules—your policy, your trade-offs.
                </p>

                <div className="flex flex-wrap gap-2">
                  <FeatureBadge label="Latency-aware" />
                  <FeatureBadge label="Cost caps" />
                  <FeatureBadge label="Compliance routes" />
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

                <Alert variant="primary">
                  <AlertDescription>Optimize for cost, latency, or compliance—your rules.</AlertDescription>
                </Alert>
              </FeatureTabContent>
            </TabsContent>

            <TabsContent value="sse" className="mt-8">
              <FeatureTabContent>
                <FeatureHeader title="Task-Based API with SSE" subtitle="Stream job lifecycle into your UI" />

                <p className="text-base md:text-lg text-muted-foreground">
                  Stream job lifecycle events—model loads, token output, cost—right into your UI.
                </p>

                <div className="flex flex-wrap gap-2">
                  <FeatureBadge label="Real-time" />
                  <FeatureBadge label="Back-pressure safe" />
                  <FeatureBadge label="Cost visible" />
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

                <Alert variant="success">
                  <AlertDescription>Full visibility into every inference job.</AlertDescription>
                </Alert>
              </FeatureTabContent>
            </TabsContent>
          </div>
        </Tabs>
      </div>
    </SectionContainer>
  )
}
