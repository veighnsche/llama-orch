<!-- TEAM-FE-004: Created TabsList subcomponent -->
<script setup lang="ts">
import { computed } from 'vue'
import { TabsList as RadixTabsList } from 'radix-vue'
import { cn } from '../../../lib/utils'

interface Props {
  class?: string
}

const props = defineProps<Props>()

const classes = computed(() =>
  cn(
    'bg-muted text-muted-foreground inline-flex h-9 w-fit items-center justify-center rounded-lg p-[3px]',
    props.class
  )
)
</script>

<template>
  <RadixTabsList :class="classes">
    <slot />
  </RadixTabsList>
</template>
