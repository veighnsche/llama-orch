<!-- TEAM-FE-006: Implemented AccordionItem component -->
<script setup lang="ts">
import { inject, computed } from 'vue'

interface Props {
  value: string
  class?: string
}

const props = defineProps<Props>()

const accordion = inject<{
  toggleItem: (value: string) => void
  isOpen: (value: string) => boolean
}>('accordion')

const isOpen = computed(() => accordion?.isOpen(props.value) || false)

const toggle = () => {
  accordion?.toggleItem(props.value)
}

// Provide to children
import { provide } from 'vue'
provide('accordionItem', { value: props.value, isOpen, toggle })
</script>

<template>
  <div :class="props.class">
    <slot />
  </div>
</template>
