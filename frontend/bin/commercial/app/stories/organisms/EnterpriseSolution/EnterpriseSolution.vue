<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-008: Implemented EnterpriseSolution -->
<script setup lang="ts">
import { Shield, Lock, FileCheck, Server } from 'lucide-vue-next'

interface SecurityLayer {
  icon: any
  name: string
  description: string
  color: string
}

interface Benefit {
  value: string
  title: string
  description: string
}

interface Props {
  title?: string
  subtitle?: string
  architectureTitle?: string
  architectureSubtitle?: string
  securityLayers?: SecurityLayer[]
  benefits?: Benefit[]
}

withDefaults(defineProps<Props>(), {
  title: 'EU-Native AI Infrastructure',
  subtitle:
    'rbee provides enterprise-grade AI infrastructure that meets your compliance requirements by design. Self-hosted, EU-resident, fully auditable.',
  architectureTitle: 'Defense-in-Depth Security Architecture',
  architectureSubtitle: 'Five specialized security layers working together',
  securityLayers: () => [
    {
      icon: Shield,
      name: 'Input Validation',
      description: 'First line of defense - Rejects malicious input, prevents injection attacks',
      color: 'text-blue-400',
    },
    {
      icon: Lock,
      name: 'Authentication (auth-min)',
      description: 'Timing-safe token validation, zero-trust principles',
      color: 'text-primary',
    },
    {
      icon: FileCheck,
      name: 'Secrets Management',
      description: 'File-based credentials, memory zeroization, no environment leakage',
      color: 'text-accent',
    },
    {
      icon: FileCheck,
      name: 'Audit Logging',
      description: 'Immutable audit trail, 7-year retention, tamper-evident',
      color: 'text-purple-400',
    },
    {
      icon: Server,
      name: 'Deadline Propagation',
      description: 'Resource enforcement, prevents exhaustion attacks',
      color: 'text-destructive',
    },
  ],
  benefits: () => [
    {
      value: '100%',
      title: 'Data Sovereignty',
      description: 'Data never leaves your infrastructure. EU-only deployment. Complete control.',
    },
    {
      value: '7 Years',
      title: 'Audit Retention',
      description: 'GDPR-compliant audit logs. Immutable, tamper-evident, legally defensible.',
    },
    {
      value: '32',
      title: 'Audit Event Types',
      description: 'Complete visibility. Authentication, data access, compliance events.',
    },
    {
      value: 'Zero',
      title: 'US Cloud Dependencies',
      description: 'Self-hosted or EU marketplace. No Schrems II concerns. Full compliance.',
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

      <!-- Architecture Diagram -->
      <div class="mb-16 rounded-lg border border-border bg-card p-8">
        <div class="mb-6 text-center">
          <h3 class="mb-2 text-2xl font-semibold text-foreground">{{ architectureTitle }}</h3>
          <p class="text-muted-foreground">{{ architectureSubtitle }}</p>
        </div>

        <div class="space-y-4">
          <div
            v-for="(layer, i) in securityLayers"
            :key="i"
            class="flex items-start gap-4 rounded-lg border border-border bg-secondary p-4"
          >
            <div :class="['flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-muted', layer.color]">
              <component :is="layer.icon" class="h-5 w-5" />
            </div>
            <div>
              <div class="mb-1 font-semibold text-foreground">{{ layer.name }}</div>
              <div class="text-sm leading-relaxed text-muted-foreground">{{ layer.description }}</div>
            </div>
          </div>
        </div>

        <div class="mt-6 text-center text-sm text-muted-foreground">
          Each layer catches what others might miss - Defense-in-depth architecture
        </div>
      </div>

      <!-- Key Benefits -->
      <div class="grid gap-8 md:grid-cols-2 lg:grid-cols-4">
        <div v-for="(benefit, i) in benefits" :key="i" class="rounded-lg border border-border bg-card p-6">
          <div class="mb-3 text-3xl font-bold text-primary">{{ benefit.value }}</div>
          <div class="mb-2 font-semibold text-foreground">{{ benefit.title }}</div>
          <div class="text-sm leading-relaxed text-muted-foreground">
            {{ benefit.description }}
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
