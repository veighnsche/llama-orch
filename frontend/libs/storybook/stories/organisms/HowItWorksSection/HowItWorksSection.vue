<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-006: Implemented HowItWorksSection component -->
<script setup lang="ts">
interface Step {
  number: number
  title: string
  description: string
  code: string[]
  reverse?: boolean
}

interface Props {
  title?: string
  steps?: Step[]
}

const props = withDefaults(defineProps<Props>(), {
  title: 'From Zero to AI Infrastructure in 15 Minutes',
  steps: () => [
    {
      number: 1,
      title: 'Install rbee',
      description: 'One command to install rbee on your machine. Works on Linux, macOS, and Windows.',
      code: [
        '$ curl -sSL https://rbee.dev/install.sh | sh',
        '$ rbee-keeper daemon start',
        '  ✓ rbee daemon started on port 8080',
      ],
    },
    {
      number: 2,
      title: 'Add Your Machines',
      description: 'Connect all your GPUs across your network. rbee automatically detects CUDA, Metal, and CPU backends.',
      code: [
        '$ rbee-keeper setup add-node \\',
        '  --name workstation \\',
        '  --ssh-host 192.168.1.10',
        '  ✓ Added node: workstation (2x RTX 4090)',
        '$ rbee-keeper setup add-node \\',
        '  --name mac \\',
        '  --ssh-host 192.168.1.20',
        '  ✓ Added node: mac (M2 Ultra)',
      ],
      reverse: true,
    },
    {
      number: 3,
      title: 'Start Inference',
      description: 'Point your tools to localhost. Zed, Cursor, or any OpenAI-compatible tool works instantly.',
      code: [
        '$ export OPENAI_API_BASE=http://localhost:8080/v1',
        '',
        '# Now Zed, Cursor, or any OpenAI-compatible',
        '# tool works with your local infrastructure!',
      ],
    },
    {
      number: 4,
      title: 'Build AI Agents',
      description: 'Use the TypeScript SDK to build custom AI agents, tools, and workflows.',
      code: [
        "import { invoke } from '@llama-orch/utils'",
        '',
        "const code = await invoke({",
        "  prompt: 'Generate API',",
        "  model: 'llama-3.1-70b'",
        "})",
      ],
      reverse: true,
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

      <div class="max-w-5xl mx-auto space-y-16">
        <div
          v-for="step in steps"
          :key="step.number"
          class="grid lg:grid-cols-2 gap-8 items-center"
        >
          <div :class="['space-y-4', step.reverse ? 'lg:order-2' : '']">
            <div class="inline-flex items-center justify-center h-12 w-12 rounded-full bg-primary text-primary-foreground font-bold text-xl">
              {{ step.number }}
            </div>
            <h3 class="text-2xl font-bold text-foreground">{{ step.title }}</h3>
            <p class="text-muted-foreground leading-relaxed">
              {{ step.description }}
            </p>
          </div>
          <div :class="['bg-card border border-border rounded-lg p-6 font-mono text-sm', step.reverse ? 'lg:order-1' : '']">
            <div v-for="(line, index) in step.code" :key="index" :class="[
              line.startsWith('$') ? 'text-green-600' : 
              line.startsWith('#') ? 'text-muted-foreground' :
              line.includes('✓') ? 'text-foreground pl-4' :
              line.startsWith('  ') ? 'text-muted-foreground' :
              line.includes('import') || line.includes('const') || line.includes('await') ? 'text-primary' :
              'text-foreground',
              index > 0 ? 'mt-2' : ''
            ]">
              {{ line }}
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
