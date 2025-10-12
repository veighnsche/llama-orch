<!-- TEAM-FE-006: Implemented AccordionTrigger component -->
<script setup lang="ts">
import { inject } from 'vue'
import { ChevronDown } from 'lucide-vue-next'

interface Props {
  class?: string
}

const props = defineProps<Props>()

const accordionItem = inject<{
  toggle: () => void
  isOpen: { value: boolean }
}>('accordionItem')
</script>

<template>
  <button
    type="button"
    :class="['flex w-full items-center justify-between py-4 text-left transition-all', props.class]"
    @click="accordionItem?.toggle()"
  >
    <slot />
    <ChevronDown
      :class="[
        'h-4 w-4 shrink-0 transition-transform duration-200',
        accordionItem?.isOpen.value ? 'rotate-180' : '',
      ]"
    />
  </button>
</template>
