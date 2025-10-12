<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-006: Implemented Accordion component -->
<script setup lang="ts">
import { ref, provide } from 'vue'

interface Props {
  type?: 'single' | 'multiple'
  collapsible?: boolean
  defaultValue?: string | string[]
}

const props = withDefaults(defineProps<Props>(), {
  type: 'single',
  collapsible: true,
})

const openItems = ref<string[]>(
  props.defaultValue
    ? Array.isArray(props.defaultValue)
      ? props.defaultValue
      : [props.defaultValue]
    : []
)

const toggleItem = (value: string) => {
  if (props.type === 'single') {
    if (openItems.value.includes(value) && props.collapsible) {
      openItems.value = []
    } else {
      openItems.value = [value]
    }
  } else {
    if (openItems.value.includes(value)) {
      openItems.value = openItems.value.filter((item) => item !== value)
    } else {
      openItems.value = [...openItems.value, value]
    }
  }
}

const isOpen = (value: string) => openItems.value.includes(value)

provide('accordion', { toggleItem, isOpen })
</script>

<template>
  <div class="space-y-4">
    <slot />
  </div>
</template>
