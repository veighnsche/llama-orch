<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-006: Implemented UseCasesSection component -->
<script setup lang="ts">
import { Building, Home, Laptop, Users } from 'lucide-vue-next'

interface UseCase {
  icon: 'Laptop' | 'Users' | 'Home' | 'Building'
  iconColor: string
  title: string
  scenario: string
  solution: string
  outcome: string
}

interface Props {
  title?: string
  useCases?: UseCase[]
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Built for Those Who Value Independence',
  useCases: () => [
    {
      icon: 'Laptop',
      iconColor: 'text-blue-600 bg-blue-100',
      title: 'The Solo Developer',
      scenario: 'Building a SaaS with AI features. Uses Claude for coding but fears vendor lock-in.',
      solution: 'Runs rbee on gaming PC + old workstation. Llama 70B for coding, Stable Diffusion for assets.',
      outcome: '$0/month AI costs. Complete control. Never blocked by rate limits.',
    },
    {
      icon: 'Users',
      iconColor: 'text-primary bg-accent',
      title: 'The Small Team',
      scenario: '5-person startup. Spending $500/month on AI APIs. Need to cut costs.',
      solution: 'Pools team\'s hardware. 3 workstations + 2 Macs = 8 GPUs total. Shared rbee cluster.',
      outcome: 'Saves $6,000/year. Faster inference. GDPR-compliant.',
    },
    {
      icon: 'Home',
      iconColor: 'text-green-600 bg-green-100',
      title: 'The Homelab Enthusiast',
      scenario: 'Has 4 GPUs collecting dust. Wants to build AI agents for personal projects.',
      solution: 'Runs rbee across homelab. Builds custom AI coder, documentation generator, code reviewer.',
      outcome: 'Turns idle hardware into productive AI infrastructure.',
    },
    {
      icon: 'Building',
      iconColor: 'text-muted-foreground bg-muted',
      title: 'The Enterprise',
      scenario: '50-person dev team. Can\'t send code to external APIs due to compliance.',
      solution: 'Deploys rbee on-premises. 20 GPUs across data center. Custom Rhai routing for compliance.',
      outcome: 'EU-only routing. Full audit trail. Zero external dependencies.',
    },
  ],
})

const getIcon = (iconName: string) => {
  const icons = { Laptop, Users, Home, Building }
  return icons[iconName as keyof typeof icons]
}
</script>

<template>
  <section class="py-24 bg-background">
    <div class="container mx-auto px-4">
      <div class="max-w-4xl mx-auto text-center mb-16">
        <h2 class="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance">
          {{ title }}
        </h2>
      </div>

      <div class="grid md:grid-cols-2 gap-8 max-w-6xl mx-auto">
        <div
          v-for="(useCase, index) in useCases"
          :key="index"
          class="bg-secondary border border-border rounded-lg p-8 space-y-4"
        >
          <div :class="['h-12 w-12 rounded-lg flex items-center justify-center', useCase.iconColor]">
            <component :is="getIcon(useCase.icon)" class="h-6 w-6" />
          </div>
          <h3 class="text-xl font-bold text-foreground">{{ useCase.title }}</h3>
          <div class="space-y-3 text-sm">
            <p class="text-muted-foreground">
              <span class="font-medium text-foreground">Scenario:</span> {{ useCase.scenario }}
            </p>
            <p class="text-muted-foreground">
              <span class="font-medium text-foreground">Solution:</span> {{ useCase.solution }}
            </p>
            <p class="text-green-700 font-medium">
              ✓ {{ useCase.outcome }}
            </p>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
