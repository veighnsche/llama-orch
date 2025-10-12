<!-- Created by: TEAM-FE-001 -->
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
    '@container/card-header grid auto-rows-min grid-rows-[auto_auto] items-start gap-2 px-6 has-data-[slot=card-action]:grid-cols-[1fr_auto] [.border-b]:pb-6',
    props.class
  )
)
</script>

<template>
  <div
    :class="classes"
    data-slot="card-header"
  >
    <slot />
  </div>
</template>
