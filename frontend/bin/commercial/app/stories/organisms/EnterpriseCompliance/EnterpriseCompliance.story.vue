<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-009: Added story variants -->
<script setup lang="ts">
import { Globe } from 'lucide-vue-next'
import EnterpriseCompliance from './EnterpriseCompliance.vue'
</script>

<template>
  <Story title="organisms/EnterpriseCompliance">
    <Variant title="Default">
      <EnterpriseCompliance />
    </Variant>

    <Variant title="GDPR Only">
      <EnterpriseCompliance
        :standards="[
          {
            icon: Globe,
            title: 'GDPR',
            subtitle: 'EU Regulation',
            items: [
              '7-year audit retention (Article 30)',
              'Data access records (Article 15)',
              'Right to erasure tracking (Article 17)'
            ],
            endpoints: [
              'GET /v2/compliance/data-access',
              'POST /v2/compliance/data-export'
            ]
          }
        ]"
      />
    </Variant>

    <Variant title="Custom CTA">
      <EnterpriseCompliance
        cta-title="Get Compliance Documentation"
        cta-subtitle="Download our complete compliance package."
        primary-cta="Download Now"
        secondary-cta="Schedule Call"
      />
    </Variant>
  </Story>
</template>
