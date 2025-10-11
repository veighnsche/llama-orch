<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-009: Added story variants -->
<script setup lang="ts">
import { Shield, Lock, FileCheck } from 'lucide-vue-next'
import EnterpriseSolution from './EnterpriseSolution.vue'
</script>

<template>
  <Story title="organisms/EnterpriseSolution">
    <Variant title="Default">
      <EnterpriseSolution />
    </Variant>

    <Variant title="Custom Security Layers (3 layers)">
      <EnterpriseSolution
        :security-layers="[
          {
            icon: Shield,
            name: 'Input Validation',
            description: 'First line of defense - Rejects malicious input',
            color: 'text-blue-400'
          },
          {
            icon: Lock,
            name: 'Authentication',
            description: 'Timing-safe token validation',
            color: 'text-primary'
          },
          {
            icon: FileCheck,
            name: 'Audit Logging',
            description: 'Immutable audit trail',
            color: 'text-purple-400'
          }
        ]"
      />
    </Variant>

    <Variant title="Custom Benefits">
      <EnterpriseSolution
        :benefits="[
          {
            value: '100%',
            title: 'Data Sovereignty',
            description: 'Complete control over your data'
          },
          {
            value: 'Zero',
            title: 'Cloud Dependencies',
            description: 'Self-hosted infrastructure'
          }
        ]"
      />
    </Variant>
  </Story>
</template>
