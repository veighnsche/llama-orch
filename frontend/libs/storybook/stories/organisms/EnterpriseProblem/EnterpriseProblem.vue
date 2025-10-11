<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-008: Implemented EnterpriseProblem -->
<script setup lang="ts">
import { AlertTriangle, Globe, FileX, Scale } from 'lucide-vue-next'

interface Problem {
  icon: any
  title: string
  description: string
}

interface Props {
  title?: string
  subtitle?: string
  problems?: Problem[]
  quoteText?: string
  quoteAuthor?: string
}

withDefaults(defineProps<Props>(), {
  title: 'The Compliance Challenge of Cloud AI',
  subtitle:
    'Using external AI providers creates compliance risks that can cost millions in fines and damage your reputation.',
  problems: () => [
    {
      icon: Globe,
      title: 'Data Sovereignty Violations',
      description:
        'Your sensitive data crosses borders to US cloud providers. GDPR Article 44 violations. Schrems II compliance impossible. Data Protection Authorities watching.',
    },
    {
      icon: FileX,
      title: 'Missing Audit Trails',
      description:
        'No immutable logs. No proof of compliance. Cannot demonstrate GDPR Article 30 compliance. SOC2 audits fail. ISO 27001 certification impossible.',
    },
    {
      icon: Scale,
      title: 'Regulatory Fines',
      description:
        'GDPR fines up to €20M or 4% of global revenue. Healthcare (HIPAA) violations: $50K per record. Financial services (PCI-DSS) breaches: reputation destroyed.',
    },
    {
      icon: AlertTriangle,
      title: 'Zero Control',
      description:
        'Provider changes terms. Data Processing Agreements worthless. Cannot guarantee data residency. Cannot prove compliance. Your DPO cannot sleep.',
    },
  ],
  quoteText: '"We cannot use external AI providers due to GDPR compliance requirements."',
  quoteAuthor: '— Every EU CTO and Data Protection Officer',
})
</script>

<template>
  <section class="border-b border-border bg-gradient-to-b from-destructive/5 via-background to-background px-6 py-24">
    <div class="mx-auto max-w-7xl">
      <div class="mb-16 text-center">
        <h2 class="mb-4 text-4xl font-bold text-foreground">{{ title }}</h2>
        <p class="mx-auto max-w-3xl text-balance text-xl text-muted-foreground">
          {{ subtitle }}
        </p>
      </div>

      <div class="grid gap-8 md:grid-cols-2 lg:grid-cols-4">
        <div
          v-for="(problem, i) in problems"
          :key="i"
          class="group rounded-lg border border-destructive/30 bg-gradient-to-b from-destructive/10 to-background p-6 transition-all hover:border-destructive/50"
        >
          <div class="mb-4 flex h-12 w-12 items-center justify-center rounded-lg bg-destructive/10">
            <component :is="problem.icon" class="h-6 w-6 text-destructive" />
          </div>
          <h3 class="mb-3 text-xl font-semibold text-foreground">{{ problem.title }}</h3>
          <p class="text-pretty leading-relaxed text-muted-foreground">
            {{ problem.description }}
          </p>
        </div>
      </div>

      <!-- Bottom emphasis -->
      <div class="mt-12 rounded-lg border border-destructive/30 bg-destructive/10 p-8 text-center">
        <p class="text-balance text-xl font-semibold text-destructive">
          {{ quoteText }}
        </p>
        <p class="mt-2 text-muted-foreground">{{ quoteAuthor }}</p>
      </div>
    </div>
  </section>
</template>
