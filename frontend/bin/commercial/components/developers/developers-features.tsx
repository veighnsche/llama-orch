"use client"

import { useState } from "react"
import { Code, Cpu, Gauge, Terminal } from "lucide-react"

const features = [
  {
    id: "openai-api",
    icon: Code,
    title: "OpenAI-Compatible API",
    description:
      "Drop-in replacement for OpenAI API. Works with Zed, Cursor, Continue, and any tool that supports OpenAI.",
    benefit: "No code changes. Just point to localhost.",
    code: `export OPENAI_API_BASE=http://localhost:8080/v1
export OPENAI_API_KEY=your-rbee-token

# Now Zed IDE uses YOUR infrastructure
zed .`,
  },
  {
    id: "multi-gpu",
    icon: Cpu,
    title: "Multi-GPU Orchestration",
    description: "Automatically distribute workloads across CUDA, Metal, and CPU backends. Use every GPU you own.",
    benefit: "10x throughput by using all your hardware.",
    code: `rbee-keeper worker start --gpu 0 --backend cuda  # PC
rbee-keeper worker start --gpu 1 --backend cuda  # PC
rbee-keeper worker start --gpu 0 --backend metal # Mac

# All GPUs work together automatically`,
  },
  {
    id: "task-api",
    icon: Terminal,
    title: "Task-Based API with SSE",
    description: "Real‑time progress updates. See model loading, token generation, and cost tracking as it happens.",
    benefit: "Full visibility into every inference job.",
    code: `event: started
data: {"queue_position":3}

event: token
data: {"t":"Hello","i":0}

event: metrics
data: {"tokens_remaining":98}`,
  },
  {
    id: "shutdown",
    icon: Gauge,
    title: "Cascading Shutdown",
    description: "Press Ctrl+C once. Everything shuts down cleanly. No orphaned processes. No leaked VRAM.",
    benefit: "Reliable cleanup. Safe for development.",
    code: `# Press Ctrl+C
^C
Shutting down queen-rbee...
Stopping all hives...
Terminating workers...
✓ Clean shutdown complete`,
  },
]

export function DevelopersFeatures() {
  const [activeTab, setActiveTab] = useState("openai-api")

  const activeFeature = features.find((f) => f.id === activeTab)!

  return (
    <section className="border-b border-border py-24">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="mb-4 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
            Enterprise-Grade Features. Homelab Simplicity.
          </h2>
        </div>

        <div className="mx-auto mt-16 max-w-6xl">
          {/* Tabs */}
          <div className="mb-8 flex flex-wrap gap-2">
            {features.map((feature) => (
              <button
                key={feature.id}
                onClick={() => setActiveTab(feature.id)}
                className={`flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium transition-all ${
                  activeTab === feature.id
                    ? "border-primary bg-primary/10 text-primary"
                    : "border-border bg-card text-muted-foreground hover:border-border hover:text-foreground"
                }`}
              >
                <feature.icon className="h-4 w-4" />
                {feature.title}
              </button>
            ))}
          </div>

          {/* Content */}
          <div className="grid gap-8 lg:grid-cols-2">
            <div>
              <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                <activeFeature.icon className="h-6 w-6 text-primary" />
              </div>
              <h3 className="mb-3 text-2xl font-semibold text-card-foreground">{activeFeature.title}</h3>
              <p className="mb-4 leading-relaxed text-muted-foreground">{activeFeature.description}</p>
              <div className="rounded-lg border border-primary/30 bg-primary/10 p-4">
                <div className="text-sm font-medium text-primary">Benefit</div>
                <div className="mt-1 text-foreground">{activeFeature.benefit}</div>
              </div>
            </div>

            <div className="overflow-hidden rounded-lg border border-border bg-card">
              <div className="border-b border-border bg-muted px-4 py-2">
                <span className="text-sm text-muted-foreground">Example</span>
              </div>
              <div className="p-4">
                <pre className="overflow-x-auto font-mono text-sm text-foreground">
                  <code>{activeFeature.code}</code>
                </pre>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
