<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-009: Added story variants -->
<script setup lang="ts">
import { Server, Shield, CheckCircle } from 'lucide-vue-next'
import EnterpriseHowItWorks from './EnterpriseHowItWorks.vue'
</script>

<template>
  <Story title="organisms/EnterpriseHowItWorks">
    <Variant title="Default">
      <EnterpriseHowItWorks />
    </Variant>

    <Variant title="3 Steps">
      <EnterpriseHowItWorks
        :steps="[
          {
            icon: Shield,
            title: 'Compliance Assessment',
            description: 'Review your compliance requirements and create a tailored deployment plan.',
            deliverables: ['Compliance gap analysis', 'Risk assessment report']
          },
          {
            icon: Server,
            title: 'On-Premises Deployment',
            description: 'Deploy rbee on your infrastructure.',
            deliverables: ['EU data centers', 'On-premises servers']
          },
          {
            icon: CheckCircle,
            title: 'Production Launch',
            description: 'Go live with enterprise SLAs.',
            deliverables: ['99.9% uptime SLA', '24/7 support']
          }
        ]"
        :timeline="[
          { duration: 'Week 1-2', label: 'Assessment' },
          { duration: 'Week 3-4', label: 'Deployment' },
          { duration: 'Week 5', label: 'Launch' }
        ]"
      />
    </Variant>

    <Variant title="Custom Timeline">
      <EnterpriseHowItWorks
        timeline-title="Fast Track Deployment"
        timeline-subtitle="Accelerated timeline for urgent projects"
      />
    </Variant>
  </Story>
</template>
