<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-009: Implemented UseCasesGrid -->
<script setup lang="ts">
import { Laptop, Users, Home, Building, Briefcase, GraduationCap, Code, Server } from 'lucide-vue-next'

interface UseCase {
  icon: any
  iconColor: string
  title: string
  scenario: string
  solution: string
  benefit: string
}

interface Props {
  title?: string
  useCases?: UseCase[]
}

withDefaults(defineProps<Props>(), {
  title: 'Real Scenarios. Real Solutions.',
  useCases: () => [
    {
      icon: Laptop,
      iconColor: 'text-blue-600 bg-blue-100',
      title: 'The Solo Developer',
      scenario: 'Building a SaaS with AI features. Uses Claude for coding but fears vendor lock-in.',
      solution: 'Runs rbee on gaming PC + old workstation. Llama 70B for coding, Stable Diffusion for assets.',
      benefit: '$0/month AI costs. Complete control. Never blocked by rate limits.',
    },
    {
      icon: Users,
      iconColor: 'text-amber-600 bg-amber-100',
      title: 'The Small Team',
      scenario: '5-person startup. Spending $500/month on AI APIs. Need to cut costs.',
      solution: 'Pools team\'s hardware. 3 workstations + 2 Macs = 8 GPUs total. Shared rbee cluster.',
      benefit: 'Saves $6,000/year. Faster inference. GDPR-compliant.',
    },
    {
      icon: Home,
      iconColor: 'text-accent bg-accent/10',
      title: 'The Homelab Enthusiast',
      scenario: 'Has 4 GPUs collecting dust. Wants to build AI agents for personal projects.',
      solution: 'Runs rbee across homelab. Builds custom AI coder, documentation generator, code reviewer.',
      benefit: 'Turns idle hardware into productive AI infrastructure.',
    },
    {
      icon: Building,
      iconColor: 'text-muted-foreground bg-muted',
      title: 'The Enterprise',
      scenario: '50-person dev team. Can\'t send code to external APIs due to compliance.',
      solution: 'Deploys rbee on-premises. 20 GPUs across data center. Custom Rhai routing for compliance.',
      benefit: 'EU-only routing. Full audit trail. Zero external dependencies.',
    },
    {
      icon: Briefcase,
      iconColor: 'text-amber-600 bg-amber-100',
      title: 'The Freelance Developer',
      scenario: 'Works on multiple client projects. Needs AI assistance but can\'t share client code with external APIs.',
      solution: 'Runs rbee locally. All client code stays on local machine. Uses Llama for code generation and review.',
      benefit: 'Client confidentiality maintained. Professional AI tools. Zero subscription costs.',
    },
    {
      icon: GraduationCap,
      iconColor: 'text-blue-600 bg-blue-100',
      title: 'The Research Lab',
      scenario: 'University research lab with grant funding. Needs AI for research but limited budget for cloud services.',
      solution: 'Deploys rbee on lab\'s GPU cluster. Uses grant money for hardware, not subscriptions. Full control over models.',
      benefit: 'Maximizes research budget. Reproducible experiments. No vendor dependency.',
    },
    {
      icon: Code,
      iconColor: 'text-accent bg-accent/10',
      title: 'The Open Source Maintainer',
      scenario: 'Maintains popular open source projects. Wants AI assistance for code reviews and documentation but can\'t afford enterprise AI tools.',
      solution: 'Runs rbee on personal hardware. Builds custom AI agents for PR reviews, documentation generation, and issue triage.',
      benefit: 'Sustainable AI tooling. Community-aligned. Zero ongoing costs.',
    },
    {
      icon: Server,
      iconColor: 'text-muted-foreground bg-muted',
      title: 'The GPU Provider',
      scenario: 'Has idle GPU hardware (former mining rig, gaming PC). Wants to monetize excess capacity.',
      solution: 'Joins rbee marketplace (M3). Sets pricing and availability. Earns passive income from idle hardware.',
      benefit: 'Passive income stream. Help the community. Control when GPUs are available.',
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

      <div class="grid md:grid-cols-2 gap-8 max-w-6xl mx-auto">
        <div v-for="(useCase, i) in useCases" :key="i" class="bg-secondary border border-border rounded-lg p-8 space-y-4">
          <div :class="['h-12 w-12 rounded-lg flex items-center justify-center', useCase.iconColor]">
            <component :is="useCase.icon" class="h-6 w-6" />
          </div>
          <h3 class="text-xl font-bold text-foreground">{{ useCase.title }}</h3>
          <div class="space-y-3 text-sm">
            <p class="text-muted-foreground">
              <span class="font-medium text-foreground">Scenario:</span> {{ useCase.scenario }}
            </p>
            <p class="text-muted-foreground">
              <span class="font-medium text-foreground">Solution:</span> {{ useCase.solution }}
            </p>
            <p class="text-accent font-medium">
              ✓ {{ useCase.benefit }}
            </p>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
