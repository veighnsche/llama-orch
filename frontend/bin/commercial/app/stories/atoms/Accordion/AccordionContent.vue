<!-- TEAM-FE-006: Implemented AccordionContent component -->
<script setup lang="ts">
import { inject } from 'vue'

interface Props {
  class?: string
}

const props = defineProps<Props>()

const accordionItem = inject<{
  isOpen: { value: boolean }
}>('accordionItem')
</script>

<template>
  <div
    v-if="accordionItem?.isOpen.value"
    :class="['pb-4 pt-0', props.class]"
  >
    <slot />
  </div>
</template>
