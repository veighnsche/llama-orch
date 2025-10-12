<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-001: Implemented Card component ported from React reference -->
<script setup lang="ts">
import { computed } from 'vue'
import { cn } from '../../../lib/utils'

// Define props interface
interface Props {
  class?: string
}

const props = defineProps<Props>()

// Compute classes - ported from React reference
const classes = computed(() =>
  cn(
    'bg-card text-card-foreground flex flex-col gap-6 rounded-xl border py-6 shadow-sm',
    props.class
  )
)
</script>

<template>
  <div
    :class="classes"
    data-slot="card"
  >
    <slot />
  </div>
</template>
