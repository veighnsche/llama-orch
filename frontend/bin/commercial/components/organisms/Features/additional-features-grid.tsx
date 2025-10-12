import { Shield, Database, Network, Terminal, Code } from "lucide-react"
import { SectionContainer, FeatureCard } from '@/components/molecules'

export function AdditionalFeaturesGrid() {
  return (
    <SectionContainer
      title="Everything You Need for AI Infrastructure"
      bgVariant="background"
    >
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-6xl mx-auto">
          <FeatureCard
            icon={Shield}
            title="Cascading Shutdown"
            description="Press Ctrl+C once. Everything shuts down cleanly. No orphaned processes. No leaked VRAM. Reliable cleanup guaranteed."
            iconColor="chart-2"
            size="md"
          />

          <FeatureCard
            icon={Database}
            title="Model Catalog"
            description="Automatic model provisioning from Hugging Face. Support for GGUF models. Llama, Mistral, Qwen, DeepSeek, and more."
            iconColor="chart-3"
            size="md"
          />

          <FeatureCard
            icon={Network}
            title="Network Orchestration"
            description="Orchestrate across your entire home network. Gaming PCs, workstations, Macs—all working together as one AI cluster."
            iconColor="primary"
            size="md"
          />

          <FeatureCard
            icon={Terminal}
            title="CLI & Web UI"
            description="Powerful CLI for automation and scripting. Beautiful Web UI for visual management. Choose your preferred workflow."
            iconColor="muted-foreground"
            size="md"
          />

          <FeatureCard
            icon={Code}
            title="TypeScript SDK"
            description="llama-orch-utils provides a TypeScript library for building AI agents. Type-safe, async/await, full IDE support."
            iconColor="primary"
            size="md"
          />

          <FeatureCard
            icon={Shield}
            title="Security First"
            description="Five specialized security crates. Defense-in-depth architecture. Timing-safe authentication. Immutable audit logs."
            iconColor="chart-2"
            size="md"
          />
        </div>
      </SectionContainer>
  )
}
