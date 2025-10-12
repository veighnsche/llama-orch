<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-001: Implemented Textarea component ported from React reference -->
<script setup lang="ts">
import { computed } from 'vue'
import { cn } from '~/lib/utils'

// Define props interface
interface Props {
  disabled?: boolean
  readonly?: boolean
  placeholder?: string
  modelValue?: string
  rows?: number
  class?: string
}

const props = withDefaults(defineProps<Props>(), {
  disabled: false,
  readonly: false,
  rows: 3,
})

const emit = defineEmits<{
  'update:modelValue': [value: string]
}>()

// Compute classes - ported from React reference
const classes = computed(() =>
  cn(
    'border-input placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-ring/50 aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive dark:bg-input/30 flex field-sizing-content min-h-16 w-full rounded-md border bg-transparent px-3 py-2 text-base shadow-xs transition-[color,box-shadow] outline-none focus-visible:ring-[3px] disabled:cursor-not-allowed disabled:opacity-50 md:text-sm',
    props.class
  )
)

const handleInput = (event: Event) => {
  const target = event.target as HTMLTextAreaElement
  emit('update:modelValue', target.value)
}
</script>

<template>
  <textarea
    :disabled="disabled"
    :readonly="readonly"
    :placeholder="placeholder"
    :value="modelValue"
    :rows="rows"
    :class="classes"
    data-slot="textarea"
    @input="handleInput"
  />
</template>
