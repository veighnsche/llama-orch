<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-001: Implemented Label component ported from React reference -->
<script setup lang="ts">
import { computed } from 'vue'
import { Label as RadixLabel } from 'radix-vue'
import { cn } from '~/lib/utils'

// Define props interface
interface Props {
  for?: string
  required?: boolean
  disabled?: boolean
  class?: string
}

const props = withDefaults(defineProps<Props>(), {
  required: false,
  disabled: false,
})

// Compute classes - ported from React reference
const classes = computed(() =>
  cn(
    'flex items-center gap-2 text-sm leading-none font-medium select-none group-data-[disabled=true]:pointer-events-none group-data-[disabled=true]:opacity-50 peer-disabled:cursor-not-allowed peer-disabled:opacity-50',
    props.class
  )
)
</script>

<template>
  <RadixLabel
    :for="for"
    :class="classes"
    data-slot="label"
  >
    <slot />
    <span v-if="required" class="text-destructive">*</span>
  </RadixLabel>
</template>
