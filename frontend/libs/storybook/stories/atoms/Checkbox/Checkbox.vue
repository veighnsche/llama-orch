<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-001: Implemented Checkbox component with Radix Vue -->
<script setup lang="ts">
import { computed } from 'vue'
import { CheckboxRoot, CheckboxIndicator } from 'radix-vue'
import { Check } from 'lucide-vue-next'
import { cn } from '../../../lib/utils'

// Define props interface
interface Props {
  checked?: boolean | 'indeterminate'
  disabled?: boolean
  required?: boolean
  name?: string
  value?: string
  class?: string
}

const props = withDefaults(defineProps<Props>(), {
  disabled: false,
  required: false,
})

const emit = defineEmits<{
  'update:checked': [value: boolean | 'indeterminate']
}>()

// Compute classes - ported from React reference
const classes = computed(() =>
  cn(
    'peer border-input dark:bg-input/30 data-[state=checked]:bg-primary data-[state=checked]:text-primary-foreground dark:data-[state=checked]:bg-primary data-[state=checked]:border-primary focus-visible:border-ring focus-visible:ring-ring/50 aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive size-4 shrink-0 rounded-[4px] border shadow-xs transition-shadow outline-none focus-visible:ring-[3px] disabled:cursor-not-allowed disabled:opacity-50',
    props.class
  )
)
</script>

<template>
  <CheckboxRoot
    :checked="checked"
    :disabled="disabled"
    :required="required"
    :name="name"
    :value="value"
    :class="classes"
    data-slot="checkbox"
    @update:checked="emit('update:checked', $event)"
  >
    <CheckboxIndicator
      data-slot="checkbox-indicator"
      class="flex items-center justify-center text-current transition-none"
    >
      <Check :size="14" />
    </CheckboxIndicator>
  </CheckboxRoot>
</template>
