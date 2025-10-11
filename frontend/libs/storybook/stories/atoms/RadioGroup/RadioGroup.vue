<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-001: Implemented RadioGroup component with Radix Vue -->
<script setup lang="ts">
import { computed } from 'vue'
import { RadioGroupRoot, RadioGroupItem, RadioGroupIndicator } from 'radix-vue'
import { Circle } from 'lucide-vue-next'
import { cn } from '../../../lib/utils'

// Define props interface
interface Props {
  modelValue?: string
  disabled?: boolean
  required?: boolean
  name?: string
  class?: string
}

const props = withDefaults(defineProps<Props>(), {
  disabled: false,
  required: false,
})

const emit = defineEmits<{
  'update:modelValue': [value: string]
}>()

// Compute classes - ported from React reference
const classes = computed(() =>
  cn('grid gap-3', props.class)
)
</script>

<template>
  <RadioGroupRoot
    :model-value="modelValue"
    :disabled="disabled"
    :required="required"
    :name="name"
    :class="classes"
    data-slot="radio-group"
    @update:model-value="emit('update:modelValue', $event)"
  >
    <slot />
  </RadioGroupRoot>
</template>
