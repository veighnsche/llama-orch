<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-009: Added story variants -->
<script setup lang="ts">
import { Lock, Eye, Shield } from 'lucide-vue-next'
import EnterpriseSecurity from './EnterpriseSecurity.vue'
</script>

<template>
  <Story title="organisms/EnterpriseSecurity">
    <Variant title="Default">
      <EnterpriseSecurity />
    </Variant>

    <Variant title="3 Crates">
      <EnterpriseSecurity
        :security-crates="[
          {
            icon: Lock,
            title: 'auth-min: Zero-Trust Authentication',
            subtitle: 'The Trickster Guardians',
            description: 'Timing-safe token comparison prevents attacks.',
            features: ['Timing-safe comparison', 'Token fingerprinting']
          },
          {
            icon: Eye,
            title: 'audit-logging: Compliance Engine',
            subtitle: 'Legally Defensible Proof',
            description: 'Immutable audit trail with 32 event types.',
            features: ['Immutable audit trail', '32 event types']
          },
          {
            icon: Shield,
            title: 'input-validation: First Line of Defense',
            subtitle: 'Trust No Input',
            description: 'Prevents injection attacks.',
            features: ['SQL injection prevention', 'Command injection prevention']
          }
        ]"
      />
    </Variant>

    <Variant title="Custom Guarantees">
      <EnterpriseSecurity
        :guarantees="[
          { value: '< 5%', label: 'Timing variance' },
          { value: '100%', label: 'Token fingerprinting' }
        ]"
      />
    </Variant>
  </Story>
</template>
