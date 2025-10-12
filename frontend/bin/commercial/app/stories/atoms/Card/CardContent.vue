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
  cn('px-6', props.class)
)
</script>

<template>
  <div
    :class="classes"
    data-slot="card-content"
  >
    <slot />
  </div>
</template>
