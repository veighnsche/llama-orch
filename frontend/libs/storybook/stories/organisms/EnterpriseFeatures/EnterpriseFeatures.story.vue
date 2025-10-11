<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-009: Added story variants -->
<script setup lang="ts">
import { Shield, Users } from 'lucide-vue-next'
import EnterpriseFeatures from './EnterpriseFeatures.vue'
</script>

<template>
  <Story title="organisms/EnterpriseFeatures">
    <Variant title="Default">
      <EnterpriseFeatures />
    </Variant>

    <Variant title="2 Features">
      <EnterpriseFeatures
        :features="[
          {
            icon: Shield,
            title: 'Enterprise SLAs',
            description: '99.9% uptime guarantee with 24/7 support.',
            items: ['99.9% uptime SLA', '24/7 support', 'Dedicated account manager']
          },
          {
            icon: Users,
            title: 'White-Label Option',
            description: 'Rebrand rbee as your own AI infrastructure platform.',
            items: ['Custom branding', 'Custom domain', 'UI customization']
          }
        ]"
      />
    </Variant>

    <Variant title="Custom Title">
      <EnterpriseFeatures
        title="Premium Features"
        subtitle="Advanced capabilities for enterprise deployments."
      />
    </Variant>
  </Story>
</template>
