<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-001: Implemented Switch component with Radix Vue -->
<script setup lang="ts">
import { computed } from 'vue'
import { SwitchRoot, SwitchThumb } from 'radix-vue'
import { cn } from '~/lib/utils'

// Define props interface
interface Props {
  checked?: boolean
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
  'update:checked': [value: boolean]
}>()

// Compute classes - ported from React reference
const classes = computed(() =>
  cn(
    'peer data-[state=checked]:bg-primary data-[state=unchecked]:bg-input focus-visible:border-ring focus-visible:ring-ring/50 dark:data-[state=unchecked]:bg-input/80 inline-flex h-[1.15rem] w-8 shrink-0 items-center rounded-full border border-transparent shadow-xs transition-all outline-none focus-visible:ring-[3px] disabled:cursor-not-allowed disabled:opacity-50',
    props.class
  )
)
</script>

<template>
  <SwitchRoot
    :checked="checked"
    :disabled="disabled"
    :required="required"
    :name="name"
    :value="value"
    :class="classes"
    data-slot="switch"
    @update:checked="emit('update:checked', $event)"
  >
    <SwitchThumb
      data-slot="switch-thumb"
      class="bg-background dark:data-[state=unchecked]:bg-foreground dark:data-[state=checked]:bg-primary-foreground pointer-events-none block size-4 rounded-full ring-0 transition-transform data-[state=checked]:translate-x-[calc(100%-2px)] data-[state=unchecked]:translate-x-0"
    />
  </SwitchRoot>
</template>
