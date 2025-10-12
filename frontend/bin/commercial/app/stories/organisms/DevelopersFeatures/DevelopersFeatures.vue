<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-005: Implemented DevelopersFeatures -->
<script setup lang="ts">
import { ref } from 'vue'
import { Code, Cpu, Gauge, Terminal } from 'lucide-vue-next'

interface Feature {
  id: string
  icon: 'code' | 'cpu' | 'gauge' | 'terminal'
  title: string
  description: string
  benefit: string
  code: string
}

interface Props {
  title?: string
  features?: Feature[]
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Enterprise-Grade Features. Homelab Simplicity.',
  features: () => [
    {
      id: 'openai-api',
      icon: 'code',
      title: 'OpenAI-Compatible API',
      description:
        'Drop-in replacement for OpenAI API. Works with Zed, Cursor, Continue, and any tool that supports OpenAI.',
      benefit: 'No code changes. Just point to localhost.',
      code: `export OPENAI_API_BASE=http://localhost:8080/v1
export OPENAI_API_KEY=your-rbee-token

# Now Zed IDE uses YOUR infrastructure
zed .`,
    },
    {
      id: 'multi-gpu',
      icon: 'cpu',
      title: 'Multi-GPU Orchestration',
      description: 'Automatically distribute workloads across CUDA, Metal, and CPU backends. Use every GPU you own.',
      benefit: '10x throughput by using all your hardware.',
      code: `rbee-keeper worker start --gpu 0 --backend cuda  # PC
rbee-keeper worker start --gpu 1 --backend cuda  # PC
rbee-keeper worker start --gpu 0 --backend metal # Mac

# All GPUs work together automatically`,
    },
    {
      id: 'task-api',
      icon: 'terminal',
      title: 'Task-Based API with SSE',
      description: 'Real-time progress updates. See model loading, token generation, and cost tracking as it happens.',
      benefit: 'Full visibility into every inference job.',
      code: `event: started
data: {"queue_position":3}

event: token
data: {"t":"Hello","i":0}

event: metrics
data: {"tokens_remaining":98}`,
    },
    {
      id: 'shutdown',
      icon: 'gauge',
      title: 'Cascading Shutdown',
      description: 'Press Ctrl+C once. Everything shuts down cleanly. No orphaned processes. No leaked VRAM.',
      benefit: 'Reliable cleanup. Safe for development.',
      code: `# Press Ctrl+C
^C
Shutting down queen-rbee...
Stopping all hives...
Terminating workers...
✓ Clean shutdown complete`,
    },
  ],
})

const activeTab = ref(props.features[0].id)

const activeFeature = computed(() => props.features.find((f) => f.id === activeTab.value)!)

const getIcon = (iconType: string) => {
  switch (iconType) {
    case 'code':
      return Code
    case 'cpu':
      return Cpu
    case 'gauge':
      return Gauge
    case 'terminal':
      return Terminal
    default:
      return Code
  }
}
</script>

<template>
  <section class="border-b border-border py-24">
    <div class="mx-auto max-w-7xl px-6 lg:px-8">
      <div class="mx-auto max-w-2xl text-center">
        <h2 class="mb-4 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
          {{ title }}
        </h2>
      </div>

      <div class="mx-auto mt-16 max-w-6xl">
        <!-- Tabs -->
        <div class="mb-8 flex flex-wrap gap-2">
          <button
            v-for="feature in features"
            :key="feature.id"
            @click="activeTab = feature.id"
            :class="[
              'flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium transition-all',
              activeTab === feature.id
                ? 'border-primary bg-primary/10 text-primary'
                : 'border-border bg-secondary/50 text-muted-foreground hover:border-border/80 hover:text-foreground',
            ]"
          >
            <component :is="getIcon(feature.icon)" class="h-4 w-4" />
            {{ feature.title }}
          </button>
        </div>

        <!-- Content -->
        <div class="grid gap-8 lg:grid-cols-2">
          <div>
            <div class="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
              <component :is="getIcon(activeFeature.icon)" class="h-6 w-6 text-primary" />
            </div>
            <h3 class="mb-3 text-2xl font-semibold text-foreground">{{ activeFeature.title }}</h3>
            <p class="mb-4 leading-relaxed text-muted-foreground">{{ activeFeature.description }}</p>
            <div class="rounded-lg border border-primary/30 bg-primary/10 p-4">
              <div class="text-sm font-medium text-primary">Benefit</div>
              <div class="mt-1 text-foreground">{{ activeFeature.benefit }}</div>
            </div>
          </div>

          <div class="overflow-hidden rounded-lg border border-border bg-secondary">
            <div class="border-b border-border bg-muted px-4 py-2">
              <span class="text-sm text-muted-foreground">Example</span>
            </div>
            <div class="p-4">
              <pre class="overflow-x-auto font-mono text-sm text-foreground"><code>{{ activeFeature.code }}</code></pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
