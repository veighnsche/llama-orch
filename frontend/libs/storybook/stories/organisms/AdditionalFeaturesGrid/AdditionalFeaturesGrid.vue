<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-008: Implemented AdditionalFeaturesGrid -->
<script setup lang="ts">
import { Shield, Database, Network, Terminal, Code } from 'lucide-vue-next'

interface Feature {
  icon: any
  iconColor: string
  title: string
  description: string
}

interface Props {
  title?: string
  features?: Feature[]
}

withDefaults(defineProps<Props>(), {
  title: 'Everything You Need for AI Infrastructure',
  features: () => [
    {
      icon: Shield,
      iconColor: 'text-blue-600 bg-blue-100',
      title: 'Cascading Shutdown',
      description:
        'Press Ctrl+C once. Everything shuts down cleanly. No orphaned processes. No leaked VRAM. Reliable cleanup guaranteed.',
    },
    {
      icon: Database,
      iconColor: 'text-accent bg-accent/10',
      title: 'Model Catalog',
      description:
        'Automatic model provisioning from Hugging Face. Support for GGUF models. Llama, Mistral, Qwen, DeepSeek, and more.',
    },
    {
      icon: Network,
      iconColor: 'text-primary bg-primary/10',
      title: 'Network Orchestration',
      description:
        'Orchestrate across your entire home network. Gaming PCs, workstations, Macs—all working together as one AI cluster.',
    },
    {
      icon: Terminal,
      iconColor: 'text-muted-foreground bg-muted',
      title: 'CLI & Web UI',
      description:
        'Powerful CLI for automation and scripting. Beautiful Web UI for visual management. Choose your preferred workflow.',
    },
    {
      icon: Code,
      iconColor: 'text-primary bg-primary/10',
      title: 'TypeScript SDK',
      description:
        'llama-orch-utils provides a TypeScript library for building AI agents. Type-safe, async/await, full IDE support.',
    },
    {
      icon: Shield,
      iconColor: 'text-blue-600 bg-blue-100',
      title: 'Security First',
      description:
        'Five specialized security crates. Defense-in-depth architecture. Timing-safe authentication. Immutable audit logs.',
    },
  ],
})
</script>

<template>
  <section class="py-24 bg-background">
    <div class="container mx-auto px-4">
      <div class="max-w-4xl mx-auto text-center mb-16">
        <h2 class="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance">
          {{ title }}
        </h2>
      </div>

      <div class="grid md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-6xl mx-auto">
        <div v-for="(feature, i) in features" :key="i" class="bg-secondary border border-border rounded-lg p-6 space-y-4">
          <div :class="['h-12 w-12 rounded-lg flex items-center justify-center', feature.iconColor]">
            <component :is="feature.icon" class="h-6 w-6" />
          </div>
          <h3 class="text-xl font-bold text-foreground">{{ feature.title }}</h3>
          <p class="text-muted-foreground text-sm leading-relaxed">
            {{ feature.description }}
          </p>
        </div>
      </div>
    </div>
  </section>
</template>
