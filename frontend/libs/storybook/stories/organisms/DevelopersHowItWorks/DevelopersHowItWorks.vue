<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-005: Implemented DevelopersHowItWorks -->
<script setup lang="ts">
interface Step {
  number: number
  title: string
  code: string[]
  language?: string
}

interface Props {
  title?: string
  steps?: Step[]
}

withDefaults(defineProps<Props>(), {
  title: 'From Zero to AI Infrastructure in 15 Minutes',
  steps: () => [
    {
      number: 1,
      title: 'Install rbee',
      code: ['curl -sSL https://rbee.dev/install.sh | sh', 'rbee-keeper daemon start'],
      language: 'terminal',
    },
    {
      number: 2,
      title: 'Add Your Machines',
      code: [
        'rbee-keeper setup add-node --name workstation --ssh-host 192.168.1.10',
        'rbee-keeper setup add-node --name mac --ssh-host 192.168.1.20',
      ],
      language: 'terminal',
    },
    {
      number: 3,
      title: 'Configure Your IDE',
      code: [
        'export OPENAI_API_BASE=http://localhost:8080/v1',
        '# Now Zed, Cursor, or any OpenAI-compatible tool works!',
      ],
      language: 'terminal',
    },
    {
      number: 4,
      title: 'Build AI Agents',
      code: [
        "import { invoke } from '@llama-orch/utils';",
        '',
        "const code = await invoke({",
        "  prompt: 'Generate API from schema',",
        "  model: 'llama-3.1-70b'",
        '});',
      ],
      language: 'TypeScript',
    },
  ],
})
</script>

<template>
  <section class="border-b border-border bg-secondary/30 py-24">
    <div class="mx-auto max-w-7xl px-6 lg:px-8">
      <div class="mx-auto max-w-2xl text-center">
        <h2 class="mb-4 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
          {{ title }}
        </h2>
      </div>

      <div class="mx-auto mt-16 max-w-4xl space-y-12">
        <div v-for="step in steps" :key="step.number" class="flex gap-6">
          <div
            class="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-lg bg-primary text-xl font-bold text-primary-foreground"
          >
            {{ step.number }}
          </div>
          <div class="flex-1">
            <h3 class="mb-3 text-xl font-semibold text-foreground">{{ step.title }}</h3>
            <div class="overflow-hidden rounded-lg border border-border bg-secondary">
              <div class="border-b border-border bg-muted px-4 py-2">
                <span class="text-sm text-muted-foreground">{{ step.language || 'code' }}</span>
              </div>
              <div class="p-4 font-mono text-sm text-foreground">
                <div v-for="(line, index) in step.code" :key="index" :class="line.startsWith('#') ? 'text-muted-foreground' : ''">
                  {{ line }}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
