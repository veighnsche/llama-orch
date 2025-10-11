<!-- TEAM-FE-004: Created TabsContent subcomponent -->
<script setup lang="ts">
import { computed } from 'vue'
import { TabsContent as RadixTabsContent } from 'radix-vue'
import { cn } from '../../../lib/utils'

interface Props {
  value: string
  class?: string
}

const props = defineProps<Props>()

const classes = computed(() =>
  cn('flex-1 outline-none', props.class)
)
</script>

<template>
  <RadixTabsContent :value="value" :class="classes">
    <slot />
  </RadixTabsContent>
</template>
