<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-001: Implemented Input component ported from React reference -->
<script setup lang="ts">
import { computed } from 'vue'
import { cn } from '~/lib/utils'

// Define props interface
interface Props {
  type?: 'text' | 'email' | 'password' | 'number' | 'search' | 'tel' | 'url'
  disabled?: boolean
  readonly?: boolean
  placeholder?: string
  modelValue?: string | number
  class?: string
}

const props = withDefaults(defineProps<Props>(), {
  type: 'text',
  disabled: false,
  readonly: false,
})

const emit = defineEmits<{
  'update:modelValue': [value: string]
}>()

// Compute classes - ported from React reference
const classes = computed(() =>
  cn(
    'file:text-foreground placeholder:text-muted-foreground selection:bg-primary selection:text-primary-foreground dark:bg-input/30 border-input h-9 w-full min-w-0 rounded-md border bg-transparent px-3 py-1 text-base shadow-xs transition-[color,box-shadow] outline-none file:inline-flex file:h-7 file:border-0 file:bg-transparent file:text-sm file:font-medium disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50 md:text-sm',
    'focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]',
    'aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive',
    props.class
  )
)

const handleInput = (event: Event) => {
  const target = event.target as HTMLInputElement
  emit('update:modelValue', target.value)
}
</script>

<template>
  <input
    :type="type"
    :disabled="disabled"
    :readonly="readonly"
    :placeholder="placeholder"
    :value="modelValue"
    :class="classes"
    data-slot="input"
    @input="handleInput"
  />
</template>
