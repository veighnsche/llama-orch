<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-008: Implemented EnterpriseFeatures -->
<script setup lang="ts">
import { Shield, Users, Wrench, Globe } from 'lucide-vue-next'

interface Feature {
  icon: any
  title: string
  description: string
  items: string[]
}

interface Props {
  title?: string
  subtitle?: string
  features?: Feature[]
}

withDefaults(defineProps<Props>(), {
  title: 'Enterprise Features',
  subtitle: 'Everything you need for enterprise-grade AI infrastructure deployment and management.',
  features: () => [
    {
      icon: Shield,
      title: 'Enterprise SLAs',
      description:
        '99.9% uptime guarantee with 24/7 support and 1-hour response time. Dedicated account manager and quarterly business reviews.',
      items: [
        '99.9% uptime SLA',
        '24/7 support (1-hour response)',
        'Dedicated account manager',
        'Quarterly business reviews',
      ],
    },
    {
      icon: Users,
      title: 'White-Label Option',
      description:
        'Rebrand rbee as your own AI infrastructure platform. Custom branding, domain, and UI customization available.',
      items: [
        'Custom branding and logo',
        'Custom domain (ai.yourcompany.com)',
        'UI customization',
        'API endpoint customization',
      ],
    },
    {
      icon: Wrench,
      title: 'Professional Services',
      description:
        'Expert consulting for deployment, integration, and optimization. Custom development and training available.',
      items: ['Deployment consulting', 'Integration support', 'Custom development', 'Team training'],
    },
    {
      icon: Globe,
      title: 'Multi-Region Support',
      description:
        'Deploy across multiple EU regions for redundancy and compliance. Automatic failover and load balancing.',
      items: ['Multi-region deployment (EU)', 'Automatic failover', 'Load balancing', 'Geo-redundancy'],
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
        <div v-for="(feature, i) in features" :key="i" class="rounded-lg border border-border bg-card p-8">
          <div class="mb-4 flex items-center gap-3">
            <div class="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
              <component :is="feature.icon" class="h-6 w-6 text-primary" />
            </div>
            <h3 class="text-xl font-bold text-foreground">{{ feature.title }}</h3>
          </div>
          <p class="mb-4 leading-relaxed text-muted-foreground">
            {{ feature.description }}
          </p>
          <ul class="space-y-2 text-sm text-muted-foreground">
            <li v-for="(item, j) in feature.items" :key="j" class="flex items-start gap-2">
              <span class="text-accent">✓</span>
              <span>{{ item }}</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  </section>
</template>
