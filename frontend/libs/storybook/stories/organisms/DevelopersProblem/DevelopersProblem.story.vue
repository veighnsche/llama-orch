<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-005: Added story variants -->
<script setup lang="ts">
import DevelopersProblem from './DevelopersProblem.vue'
</script>

<template>
  <Story title="organisms/DevelopersProblem">
    <Variant title="Default">
      <DevelopersProblem />
    </Variant>

    <Variant title="Custom Content">
      <DevelopersProblem
        title="The Risks of Cloud Dependency"
        subtitle="Building on someone else's infrastructure means accepting their terms, their pricing, and their timeline."
        warning-text="Your business depends on infrastructure you don't control."
      />
    </Variant>

    <Variant title="Two Problems">
      <DevelopersProblem
        :problems="[
          {
            icon: 'alert',
            title: 'Vendor Lock-In',
            description: 'Once you build on their platform, switching becomes nearly impossible.',
            borderColor: 'border-destructive/50',
            bgGradient: 'from-destructive/50 to-secondary',
            iconBg: 'bg-destructive/10',
            iconColor: 'text-destructive',
          },
          {
            icon: 'dollar',
            title: 'Unpredictable Costs',
            description: 'Usage-based pricing means your costs can spike without warning.',
            borderColor: 'border-accent/50',
            bgGradient: 'from-accent/50 to-secondary',
            iconBg: 'bg-accent/10',
            iconColor: 'text-accent',
          },
        ]"
      />
    </Variant>
  </Story>
</template>
