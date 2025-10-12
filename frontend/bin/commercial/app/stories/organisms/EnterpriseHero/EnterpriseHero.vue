<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-008: Implemented EnterpriseHero -->
<script setup lang="ts">
import { Button } from 'rbee-storybook/stories'
import { Shield, Lock, FileCheck } from 'lucide-vue-next'

interface Props {
  badge?: string
  title?: string
  description?: string
  stat1Value?: string
  stat1Label?: string
  stat2Value?: string
  stat2Label?: string
  stat3Value?: string
  stat3Label?: string
  primaryCta?: string
  secondaryCta?: string
  trust1?: string
  trust2?: string
  trust3?: string
  showAuditLog?: boolean
}

withDefaults(defineProps<Props>(), {
  badge: 'EU-Native AI Infrastructure',
  title: 'AI Infrastructure That Meets Your Compliance Requirements',
  description:
    'GDPR-compliant by design. SOC2 ready. ISO 27001 aligned. Build AI infrastructure on your terms with complete data sovereignty, immutable audit trails, and enterprise-grade security.',
  stat1Value: '100%',
  stat1Label: 'GDPR Compliant',
  stat2Value: '7 Years',
  stat2Label: 'Audit Retention',
  stat3Value: 'Zero',
  stat3Label: 'US Cloud Deps',
  primaryCta: 'Schedule Demo',
  secondaryCta: 'View Compliance Details',
  trust1: 'GDPR Compliant',
  trust2: 'SOC2 Ready',
  trust3: 'ISO 27001 Aligned',
  showAuditLog: true,
})

const auditLogs = [
  { event: 'auth.success', user: 'admin@company.eu', time: '2025-10-11 14:23:15', status: 'success' },
  { event: 'data.access', user: 'analyst@company.eu', time: '2025-10-11 14:22:48', status: 'success' },
  { event: 'task.submitted', user: 'dev@company.eu', time: '2025-10-11 14:21:33', status: 'success' },
  { event: 'compliance.export', user: 'dpo@company.eu', time: '2025-10-11 14:20:12', status: 'success' },
]
</script>

<template>
  <section class="relative overflow-hidden border-b border-border bg-gradient-to-b from-background via-secondary to-background px-6 py-24 lg:py-32">
    <div class="mx-auto max-w-7xl">
      <div class="grid gap-12 lg:grid-cols-2 lg:gap-16">
        <!-- Left: Messaging -->
        <div class="flex flex-col justify-center">
          <div class="mb-6 inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/10 px-4 py-2 text-sm text-primary">
            <Shield class="h-4 w-4" />
            <span>{{ badge }}</span>
          </div>

          <h1 class="mb-6 text-balance text-5xl font-bold leading-tight text-foreground lg:text-6xl">
            {{ title }}
          </h1>

          <p class="mb-8 text-pretty text-xl leading-relaxed text-muted-foreground">
            {{ description }}
          </p>

          <!-- Key Stats -->
          <div class="mb-8 grid grid-cols-3 gap-4">
            <div class="rounded-lg border border-border bg-card p-4">
              <div class="mb-1 text-2xl font-bold text-primary">{{ stat1Value }}</div>
              <div class="text-sm text-muted-foreground">{{ stat1Label }}</div>
            </div>
            <div class="rounded-lg border border-border bg-card p-4">
              <div class="mb-1 text-2xl font-bold text-primary">{{ stat2Value }}</div>
              <div class="text-sm text-muted-foreground">{{ stat2Label }}</div>
            </div>
            <div class="rounded-lg border border-border bg-card p-4">
              <div class="mb-1 text-2xl font-bold text-primary">{{ stat3Value }}</div>
              <div class="text-sm text-muted-foreground">{{ stat3Label }}</div>
            </div>
          </div>

          <!-- CTAs -->
          <div class="flex flex-wrap gap-4">
            <Button size="lg" class="bg-primary text-primary-foreground hover:bg-primary/90">
              {{ primaryCta }}
            </Button>
            <Button size="lg" variant="outline" class="border-border text-foreground hover:bg-secondary bg-transparent">
              {{ secondaryCta }}
            </Button>
          </div>

          <!-- Trust Indicators -->
          <div class="mt-8 flex flex-wrap items-center gap-6 text-sm text-muted-foreground">
            <div class="flex items-center gap-2">
              <FileCheck class="h-4 w-4 text-primary" />
              <span>{{ trust1 }}</span>
            </div>
            <div class="flex items-center gap-2">
              <Shield class="h-4 w-4 text-primary" />
              <span>{{ trust2 }}</span>
            </div>
            <div class="flex items-center gap-2">
              <Lock class="h-4 w-4 text-primary" />
              <span>{{ trust3 }}</span>
            </div>
          </div>
        </div>

        <!-- Right: Visual -->
        <div v-if="showAuditLog" class="flex items-center justify-center">
          <div class="relative w-full max-w-lg">
            <!-- Audit Log Mockup -->
            <div class="rounded-lg border border-border bg-card p-6 shadow-2xl">
              <div class="mb-4 flex items-center justify-between">
                <div class="flex items-center gap-2">
                  <Lock class="h-5 w-5 text-primary" />
                  <span class="font-semibold text-foreground">Immutable Audit Trail</span>
                </div>
                <div class="rounded-full bg-accent/20 px-3 py-1 text-xs text-accent-foreground">Compliant</div>
              </div>

              <!-- Audit Events -->
              <div class="space-y-3">
                <div v-for="(log, i) in auditLogs" :key="i" class="rounded border border-border bg-secondary p-3">
                  <div class="mb-1 flex items-center justify-between">
                    <span class="font-mono text-sm text-primary">{{ log.event }}</span>
                    <span class="rounded bg-accent/20 px-2 py-0.5 text-xs text-accent-foreground">{{ log.status }}</span>
                  </div>
                  <div class="text-xs text-muted-foreground">
                    <div>{{ log.user }}</div>
                    <div class="text-muted-foreground/70">{{ log.time }} UTC</div>
                  </div>
                </div>
              </div>

              <!-- Footer -->
              <div class="mt-4 flex items-center justify-between border-t border-border pt-4 text-xs text-muted-foreground">
                <span>Retention: 7 years (GDPR)</span>
                <span class="flex items-center gap-1">
                  <Shield class="h-3 w-3" />
                  Tamper-evident
                </span>
              </div>
            </div>

            <!-- Floating badges -->
            <div class="absolute -right-4 -top-4 rounded-lg border border-primary/20 bg-card px-4 py-2 shadow-lg">
              <div class="text-xs text-muted-foreground">Data Residency</div>
              <div class="font-semibold text-primary">🇪🇺 EU Only</div>
            </div>

            <div class="absolute -bottom-4 -left-4 rounded-lg border border-primary/20 bg-card px-4 py-2 shadow-lg">
              <div class="text-xs text-muted-foreground">Audit Events</div>
              <div class="font-semibold text-primary">32 Types</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
