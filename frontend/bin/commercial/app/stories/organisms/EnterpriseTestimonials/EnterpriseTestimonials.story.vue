<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-009: Added story variants -->
<script setup lang="ts">
import { Building2, Heart } from 'lucide-vue-next'
import EnterpriseTestimonials from './EnterpriseTestimonials.vue'
</script>

<template>
  <Story title="organisms/EnterpriseTestimonials">
    <Variant title="Default">
      <EnterpriseTestimonials />
    </Variant>

    <Variant title="Custom Metrics">
      <EnterpriseTestimonials
        :metrics="[
          { value: '99.9%', label: 'Uptime SLA' },
          { value: '< 1hr', label: 'Response Time' }
        ]"
      />
    </Variant>

    <Variant title="2 Testimonials">
      <EnterpriseTestimonials
        :testimonials="[
          {
            icon: Building2,
            name: 'Dr. Klaus M.',
            role: 'CTO, European Bank',
            quote: 'rbee gave us complete control with on-premises deployment.'
          },
          {
            icon: Heart,
            name: 'Anna S.',
            role: 'DPO, Healthcare Provider',
            quote: 'HIPAA and GDPR compliance were non-negotiable. rbee delivered.'
          }
        ]"
      />
    </Variant>
  </Story>
</template>
