<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-009: Added story variants -->
<script setup lang="ts">
import { Building2, Heart } from 'lucide-vue-next'
import EnterpriseUseCases from './EnterpriseUseCases.vue'
</script>

<template>
  <Story title="organisms/EnterpriseUseCases">
    <Variant title="Default">
      <EnterpriseUseCases />
    </Variant>

    <Variant title="2 Use Cases">
      <EnterpriseUseCases
        :use-cases="[
          {
            icon: Building2,
            title: 'Financial Services',
            subtitle: 'Banks, Insurance, FinTech',
            description: 'A European bank needed AI-powered code generation.',
            challenges: ['Cannot send code to external APIs (PCI-DSS)', 'Need complete audit trail (SOC2)'],
            solutions: ['On-premises deployment', 'Immutable audit logs']
          },
          {
            icon: Heart,
            title: 'Healthcare',
            subtitle: 'Hospitals, MedTech, Pharma',
            description: 'A healthcare provider wanted AI-assisted development.',
            challenges: ['HIPAA compliance (PHI protection)', 'GDPR Article 9 (health data)'],
            solutions: ['Self-hosted deployment', 'EU-only deployment']
          }
        ]"
      />
    </Variant>

    <Variant title="Custom Title">
      <EnterpriseUseCases
        title="Industry Solutions"
        subtitle="Tailored for regulated industries."
      />
    </Variant>
  </Story>
</template>
