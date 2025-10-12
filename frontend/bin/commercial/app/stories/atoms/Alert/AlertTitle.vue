<!-- Created by: TEAM-FE-001 -->
<script setup lang="ts">
import { computed } from 'vue'
import { cn } from '~/lib/utils'

// Define props interface
interface Props {
  class?: string
}

const props = defineProps<Props>()

// Compute classes - ported from React reference
const classes = computed(() =>
  cn(
    'col-start-2 line-clamp-1 min-h-4 font-medium tracking-tight',
    props.class
  )
)
</script>

<template>
  <div
    :class="classes"
    data-slot="alert-title"
  >
    <slot />
  </div>
</template>
