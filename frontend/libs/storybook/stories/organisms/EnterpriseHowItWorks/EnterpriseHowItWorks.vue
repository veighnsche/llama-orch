<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-008: Implemented EnterpriseHowItWorks -->
<script setup lang="ts">
import { Server, Shield, CheckCircle, Rocket } from 'lucide-vue-next'

interface Step {
  icon: any
  title: string
  description: string
  deliverables: string[]
}

interface TimelinePhase {
  duration: string
  label: string
}

interface Props {
  title?: string
  subtitle?: string
  steps?: Step[]
  timelineTitle?: string
  timelineSubtitle?: string
  timeline?: TimelinePhase[]
}

withDefaults(defineProps<Props>(), {
  title: 'Enterprise Deployment Process',
  subtitle:
    'From initial consultation to production deployment, we guide you through every step of the compliance journey.',
  steps: () => [
    {
      icon: Shield,
      title: 'Compliance Assessment',
      description:
        'We review your compliance requirements (GDPR, SOC2, ISO 27001, HIPAA, PCI-DSS) and create a tailored deployment plan. Identify data residency requirements, audit retention policies, and security controls.',
      deliverables: [
        'Compliance gap analysis',
        'Data flow mapping',
        'Risk assessment report',
        'Deployment architecture proposal',
      ],
    },
    {
      icon: Server,
      title: 'On-Premises Deployment',
      description:
        'Deploy rbee on your infrastructure (EU data centers, on-premises servers, or private cloud). Configure EU-only worker filtering, audit logging, and security controls. White-label option available.',
      deliverables: [
        'EU data centers (Frankfurt, Amsterdam, Paris)',
        'On-premises (your servers)',
        'Private cloud (AWS EU, Azure EU, GCP EU)',
        'Hybrid (on-prem + marketplace)',
      ],
    },
    {
      icon: CheckCircle,
      title: 'Compliance Validation',
      description:
        'Validate compliance controls with your auditors. Provide audit trail access, compliance documentation, and security architecture review. Support for SOC2 Type II, ISO 27001, and GDPR audits.',
      deliverables: [
        'Compliance documentation package',
        'Auditor access to audit logs',
        'Security architecture review',
        'Penetration testing reports',
      ],
    },
    {
      icon: Rocket,
      title: 'Production Launch',
      description:
        'Go live with enterprise SLAs, 24/7 support, and dedicated account management. Continuous monitoring, health checks, and compliance reporting. Scale as your organization grows.',
      deliverables: [
        '99.9% uptime SLA',
        '24/7 support (1-hour response time)',
        'Dedicated account manager',
        'Quarterly compliance reviews',
      ],
    },
  ],
  timelineTitle: 'Typical Deployment Timeline',
  timelineSubtitle: 'From initial consultation to production deployment',
  timeline: () => [
    { duration: 'Week 1-2', label: 'Compliance Assessment' },
    { duration: 'Week 3-4', label: 'Deployment & Configuration' },
    { duration: 'Week 5-6', label: 'Compliance Validation' },
    { duration: 'Week 7', label: 'Production Launch' },
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

      <div class="space-y-8">
        <div v-for="(step, i) in steps" :key="i" class="flex gap-6">
          <div class="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-primary text-xl font-bold text-primary-foreground">
            {{ i + 1 }}
          </div>
          <div class="flex-1">
            <div class="mb-4 flex items-center gap-3">
              <component :is="step.icon" class="h-6 w-6 text-primary" />
              <h3 class="text-2xl font-semibold text-foreground">{{ step.title }}</h3>
            </div>
            <p class="mb-4 leading-relaxed text-muted-foreground">
              {{ step.description }}
            </p>
            <div class="rounded-lg border border-border bg-card p-4">
              <div class="mb-2 font-semibold text-foreground">{{ i === 0 || i === 2 ? 'Deliverables:' : i === 1 ? 'Deployment Options:' : 'Enterprise Support:' }}</div>
              <ul class="space-y-1 text-sm text-muted-foreground">
                <li v-for="(item, j) in step.deliverables" :key="j">• {{ item }}</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      <!-- Timeline -->
      <div class="mt-12 rounded-lg border border-primary/20 bg-primary/5 p-8 text-center">
        <h3 class="mb-2 text-2xl font-semibold text-foreground">{{ timelineTitle }}</h3>
        <p class="mb-6 text-muted-foreground">{{ timelineSubtitle }}</p>
        <div class="grid gap-4 md:grid-cols-4">
          <div v-for="(phase, i) in timeline" :key="i">
            <div class="mb-2 text-3xl font-bold text-primary">{{ phase.duration }}</div>
            <div class="text-sm text-muted-foreground">{{ phase.label }}</div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
