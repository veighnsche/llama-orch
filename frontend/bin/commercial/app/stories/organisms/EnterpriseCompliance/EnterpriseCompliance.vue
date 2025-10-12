<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-008: Implemented EnterpriseCompliance -->
<script setup lang="ts">
import { Button } from '~/stories'
import { FileCheck, Shield, Lock, Globe } from 'lucide-vue-next'

interface ComplianceStandard {
  icon: any
  title: string
  subtitle: string
  items: string[]
  endpoints?: string[]
  criteria?: string[]
}

interface Props {
  title?: string
  subtitle?: string
  standards?: ComplianceStandard[]
  ctaTitle?: string
  ctaSubtitle?: string
  primaryCta?: string
  secondaryCta?: string
}

withDefaults(defineProps<Props>(), {
  title: 'Compliance by Design',
  subtitle:
    'rbee is built from the ground up to meet GDPR, SOC2, and ISO 27001 requirements. Not bolted on as an afterthought.',
  standards: () => [
    {
      icon: Globe,
      title: 'GDPR',
      subtitle: 'EU Regulation',
      items: [
        '7-year audit retention (Article 30)',
        'Data access records (Article 15)',
        'Right to erasure tracking (Article 17)',
        'Consent management (Article 7)',
        'Data residency controls (Article 44)',
        'Breach notification (Article 33)',
      ],
      endpoints: [
        'GET /v2/compliance/data-access',
        'POST /v2/compliance/data-export',
        'POST /v2/compliance/data-deletion',
        'GET /v2/compliance/audit-trail',
      ],
    },
    {
      icon: Shield,
      title: 'SOC2',
      subtitle: 'US Standard',
      items: [
        'Auditor access (query API)',
        'Security event logging (32 types)',
        '7-year retention (Type II)',
        'Tamper-evident storage (hash chains)',
        'Access control logging',
        'Encryption at rest',
      ],
      criteria: ['✓ Security (CC1-CC9)', '✓ Availability (A1.1-A1.3)', '✓ Confidentiality (C1.1-C1.2)'],
    },
    {
      icon: Lock,
      title: 'ISO 27001',
      subtitle: 'International',
      items: [
        'Security incident records (A.16)',
        '3-year retention (minimum)',
        'Access control logging (A.9)',
        'Cryptographic controls (A.10)',
        'Operations security (A.12)',
        'Information security policies (A.5)',
      ],
      criteria: ['✓ 114 controls implemented', '✓ Risk assessment framework', '✓ Continuous monitoring'],
    },
  ],
  ctaTitle: 'Ready for Your Compliance Audit',
  ctaSubtitle: 'Download our compliance documentation package or schedule a call with our compliance team.',
  primaryCta: 'Download Compliance Pack',
  secondaryCta: 'Talk to Compliance Team',
})
</script>

<template>
  <section id="compliance" class="border-b border-border bg-background px-6 py-24">
    <div class="mx-auto max-w-7xl">
      <div class="mb-16 text-center">
        <h2 class="mb-4 text-4xl font-bold text-foreground">{{ title }}</h2>
        <p class="mx-auto max-w-3xl text-balance text-xl text-muted-foreground">
          {{ subtitle }}
        </p>
      </div>

      <div class="grid gap-8 lg:grid-cols-3">
        <div v-for="(standard, i) in standards" :key="i" class="rounded-lg border border-border bg-card p-8">
          <div class="mb-6 flex items-center gap-3">
            <div class="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
              <component :is="standard.icon" class="h-6 w-6 text-primary" />
            </div>
            <div>
              <h3 class="text-2xl font-bold text-foreground">{{ standard.title }}</h3>
              <p class="text-sm text-muted-foreground">{{ standard.subtitle }}</p>
            </div>
          </div>

          <div class="space-y-3">
            <div v-for="(item, j) in standard.items" :key="j" class="flex items-start gap-2">
              <FileCheck class="mt-0.5 h-4 w-4 shrink-0 text-accent" />
              <span class="text-sm leading-relaxed text-muted-foreground">{{ item }}</span>
            </div>
          </div>

          <div v-if="standard.endpoints" class="mt-6 rounded-lg border border-accent/30 bg-accent/10 p-4">
            <div class="mb-1 font-semibold text-accent-foreground">Compliance Endpoints</div>
            <div class="space-y-1 text-xs text-muted-foreground">
              <div v-for="(endpoint, j) in standard.endpoints" :key="j">{{ endpoint }}</div>
            </div>
          </div>

          <div v-if="standard.criteria" class="mt-6 rounded-lg border border-accent/30 bg-accent/10 p-4">
            <div class="mb-1 font-semibold text-accent-foreground">{{ standard.title === 'SOC2' ? 'Trust Service Criteria' : 'ISMS Controls' }}</div>
            <div class="space-y-1 text-xs text-muted-foreground">
              <div v-for="(criterion, j) in standard.criteria" :key="j">{{ criterion }}</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Bottom CTA -->
      <div class="mt-12 rounded-lg border border-primary/20 bg-primary/5 p-8 text-center">
        <h3 class="mb-2 text-2xl font-semibold text-foreground">{{ ctaTitle }}</h3>
        <p class="mb-6 text-muted-foreground">
          {{ ctaSubtitle }}
        </p>
        <div class="flex flex-wrap justify-center gap-4">
          <Button size="lg" class="bg-primary text-primary-foreground hover:bg-primary/90">
            {{ primaryCta }}
          </Button>
          <Button size="lg" variant="outline" class="border-border text-foreground hover:bg-secondary">
            {{ secondaryCta }}
          </Button>
        </div>
      </div>
    </div>
  </section>
</template>
