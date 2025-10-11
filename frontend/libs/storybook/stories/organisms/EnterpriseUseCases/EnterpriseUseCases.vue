<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-008: Implemented EnterpriseUseCases -->
<script setup lang="ts">
import { Building2, Heart, Scale, Shield } from 'lucide-vue-next'

interface UseCase {
  icon: any
  title: string
  subtitle: string
  description: string
  challenges: string[]
  solutions: string[]
}

interface Props {
  title?: string
  subtitle?: string
  useCases?: UseCase[]
}

withDefaults(defineProps<Props>(), {
  title: 'Built for Regulated Industries',
  subtitle:
    'Organizations in highly regulated industries trust rbee for AI infrastructure that meets their compliance requirements.',
  useCases: () => [
    {
      icon: Building2,
      title: 'Financial Services',
      subtitle: 'Banks, Insurance, FinTech',
      description:
        'A European bank needed AI-powered code generation for internal tools but couldn\'t use external AI providers due to PCI-DSS and GDPR requirements.',
      challenges: [
        'Cannot send code to external APIs (PCI-DSS)',
        'Need complete audit trail (SOC2)',
        'EU data residency required (GDPR)',
        '7-year audit retention mandatory',
      ],
      solutions: [
        'On-premises deployment (EU data center)',
        'Immutable audit logs (7-year retention)',
        'Zero external dependencies',
        'SOC2 Type II compliant',
      ],
    },
    {
      icon: Heart,
      title: 'Healthcare',
      subtitle: 'Hospitals, MedTech, Pharma',
      description:
        'A healthcare provider wanted AI-assisted development for patient management systems but faced strict HIPAA and GDPR compliance requirements.',
      challenges: [
        'HIPAA compliance (PHI protection)',
        'GDPR Article 9 (health data)',
        'Cannot use US cloud providers',
        'Need breach notification capability',
      ],
      solutions: [
        'Self-hosted (hospital data center)',
        'EU-only deployment',
        'Complete audit trail (breach detection)',
        'HIPAA-compliant architecture',
      ],
    },
    {
      icon: Scale,
      title: 'Legal Services',
      subtitle: 'Law Firms, Legal Tech',
      description:
        'A law firm needed AI-powered document analysis and code generation but couldn\'t risk client confidentiality by using external AI providers.',
      challenges: [
        'Attorney-client privilege (confidentiality)',
        'Cannot send documents to external APIs',
        'Need complete audit trail (legal hold)',
        'EU data residency (GDPR)',
      ],
      solutions: [
        'On-premises deployment (law firm servers)',
        'Zero external data transfer',
        'Immutable audit logs (legal hold)',
        'Complete client confidentiality',
      ],
    },
    {
      icon: Shield,
      title: 'Government',
      subtitle: 'Public Sector, Defense',
      description:
        'A government agency needed AI-assisted development for citizen services but faced strict data sovereignty and security requirements.',
      challenges: [
        'National security (data sovereignty)',
        'Cannot use foreign cloud providers',
        'Need complete audit trail (transparency)',
        'ISO 27001 certification required',
      ],
      solutions: [
        'On-premises deployment (government data center)',
        'EU-only infrastructure',
        'ISO 27001 aligned',
        'Complete data sovereignty',
      ],
    },
  ],
})
</script>

<template>
  <section class="border-b border-border bg-background px-6 py-24">
    <div class="mx-auto max-w-7xl">
      <div class="mb-16 text-center">
        <h2 class="mb-4 text-4xl font-bold text-foreground">{{ title }}</h2>
        <p class="mx-auto max-w-3xl text-balance text-xl text-muted-foreground">
          {{ subtitle }}
        </p>
      </div>

      <div class="grid gap-8 md:grid-cols-2">
        <div v-for="(useCase, i) in useCases" :key="i" class="rounded-lg border border-border bg-card p-8">
          <div class="mb-4 flex items-center gap-3">
            <div class="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
              <component :is="useCase.icon" class="h-6 w-6 text-primary" />
            </div>
            <div>
              <h3 class="text-xl font-bold text-foreground">{{ useCase.title }}</h3>
              <p class="text-sm text-muted-foreground">{{ useCase.subtitle }}</p>
            </div>
          </div>

          <p class="mb-4 leading-relaxed text-muted-foreground">
            {{ useCase.description }}
          </p>

          <div class="mb-4 rounded-lg border border-border bg-secondary p-4">
            <div class="mb-2 font-semibold text-foreground">Challenge:</div>
            <ul class="space-y-1 text-sm text-muted-foreground">
              <li v-for="(challenge, j) in useCase.challenges" :key="j">• {{ challenge }}</li>
            </ul>
          </div>

          <div class="rounded-lg border border-accent/30 bg-accent/10 p-4">
            <div class="mb-2 font-semibold text-accent-foreground">Solution with rbee:</div>
            <ul class="space-y-1 text-sm text-muted-foreground">
              <li v-for="(solution, j) in useCase.solutions" :key="j">• {{ solution }}</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
