<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-006: Implemented TechnicalSection component -->
<script setup lang="ts">
import { Github } from 'lucide-vue-next'
import { Button } from '~/stories'

interface ArchitectureItem {
  title: string
  description: string
}

interface TechItem {
  name: string
  description: string
}

interface Props {
  title?: string
  architectureItems?: ArchitectureItem[]
  techStack?: TechItem[]
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Built by Engineers, for Engineers',
  architectureItems: () => [
    {
      title: 'BDD-Driven Development',
      description: '42/62 scenarios passing (68% complete)',
    },
    {
      title: 'Cascading Shutdown Guarantee',
      description: 'No orphaned processes, clean VRAM lifecycle',
    },
    {
      title: 'Process Isolation',
      description: 'Clean VRAM lifecycle, no memory leaks',
    },
    {
      title: 'Protocol-Aware Orchestration',
      description: 'SSE, JSON, binary protocols supported',
    },
    {
      title: 'Smart/Dumb Separation',
      description: 'Centralized intelligence, distributed execution',
    },
  ],
  techStack: () => [
    { name: 'Rust', description: 'Performance + safety' },
    { name: 'Candle ML Framework', description: 'Rust-native ML inference' },
    { name: 'Rhai Scripting', description: 'Embedded, sandboxed scripting' },
    { name: 'SQLite', description: 'Embedded database' },
    { name: 'Axum + Vue.js', description: 'Async web framework + modern UI' },
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

      <div class="grid md:grid-cols-2 gap-12 max-w-5xl mx-auto">
        <!-- Architecture Highlights -->
        <div class="space-y-6">
          <h3 class="text-2xl font-bold text-foreground">Architecture Highlights</h3>
          <ul class="space-y-3">
            <li
              v-for="(item, index) in architectureItems"
              :key="index"
              class="flex items-start gap-3"
            >
              <div class="h-6 w-6 rounded-full bg-green-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                <div class="h-2 w-2 rounded-full bg-green-600"></div>
              </div>
              <div>
                <div class="font-medium text-foreground">{{ item.title }}</div>
                <div class="text-sm text-muted-foreground">{{ item.description }}</div>
              </div>
            </li>
          </ul>
        </div>

        <!-- Technology Stack -->
        <div class="space-y-6">
          <h3 class="text-2xl font-bold text-foreground">Technology Stack</h3>
          <div class="space-y-3">
            <div
              v-for="(tech, index) in techStack"
              :key="index"
              class="bg-secondary border border-border rounded-lg p-4"
            >
              <div class="font-medium text-foreground">{{ tech.name }}</div>
              <div class="text-sm text-muted-foreground">{{ tech.description }}</div>
            </div>
          </div>

          <div class="bg-accent border border-border rounded-lg p-4 flex items-center justify-between">
            <div>
              <div class="font-bold text-accent-foreground">100% Open Source</div>
              <div class="text-sm text-muted-foreground">MIT License</div>
            </div>
            <Button variant="outline" size="sm">
              <Github class="h-4 w-4 mr-2" />
              View Source
            </Button>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
