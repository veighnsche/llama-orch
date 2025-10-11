<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-008: Implemented EnterpriseTestimonials -->
<script setup lang="ts">
import { Building2, Heart, Scale } from 'lucide-vue-next'

interface Metric {
  value: string
  label: string
}

interface Testimonial {
  icon: any
  name: string
  role: string
  quote: string
}

interface Props {
  title?: string
  subtitle?: string
  metrics?: Metric[]
  testimonials?: Testimonial[]
}

withDefaults(defineProps<Props>(), {
  title: 'Trusted by Regulated Industries',
  subtitle: 'Organizations in highly regulated industries trust rbee for compliance-first AI infrastructure.',
  metrics: () => [
    { value: '100%', label: 'GDPR Compliant' },
    { value: '7 Years', label: 'Audit Retention' },
    { value: 'Zero', label: 'Compliance Violations' },
    { value: '24/7', label: 'Enterprise Support' },
  ],
  testimonials: () => [
    {
      icon: Building2,
      name: 'Dr. Klaus M.',
      role: 'CTO, European Bank',
      quote:
        'We couldn't use external AI providers due to PCI-DSS requirements. rbee gave us complete control with on-premises deployment and immutable audit logs. SOC2 audit passed with zero findings.',
    },
    {
      icon: Heart,
      name: 'Anna S.',
      role: 'DPO, Healthcare Provider',
      quote:
        'HIPAA and GDPR compliance were non-negotiable. rbee's EU-only deployment and 7-year audit retention gave us the confidence to use AI for patient management systems. Complete data sovereignty.',
    },
    {
      icon: Scale,
      name: 'Michael R.',
      role: 'Managing Partner, Law Firm',
      quote:
        'Attorney-client privilege meant we couldn't risk external AI providers. rbee's on-premises deployment and zero external data transfer gave us the security we needed. Client confidentiality protected.',
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

      <!-- Metrics -->
      <div class="mb-16 grid gap-8 md:grid-cols-4">
        <div v-for="(metric, i) in metrics" :key="i" class="text-center">
          <div class="mb-2 text-4xl font-bold text-primary">{{ metric.value }}</div>
          <div class="text-sm text-muted-foreground">{{ metric.label }}</div>
        </div>
      </div>

      <!-- Testimonials -->
      <div class="grid gap-8 md:grid-cols-3">
        <div v-for="(testimonial, i) in testimonials" :key="i" class="rounded-lg border border-border bg-card p-8">
          <div class="mb-4 flex items-center gap-3">
            <div class="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
              <component :is="testimonial.icon" class="h-6 w-6 text-primary" />
            </div>
            <div>
              <div class="font-semibold text-foreground">{{ testimonial.name }}</div>
              <div class="text-sm text-muted-foreground">{{ testimonial.role }}</div>
            </div>
          </div>
          <p class="leading-relaxed text-muted-foreground">
            "{{ testimonial.quote }}"
          </p>
        </div>
      </div>
    </div>
  </section>
</template>
