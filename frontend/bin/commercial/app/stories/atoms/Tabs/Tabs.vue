<!-- Created by: TEAM-FE-000 (Scaffolding) -->
<!-- TEAM-FE-004: Implemented Tabs atom - ported from React reference -->
<script setup lang="ts">
import { computed } from 'vue'
import { TabsRoot } from 'radix-vue'
import { cn } from '~/lib/utils'

interface TabsRootProps {
  defaultValue?: string
  modelValue?: string
  orientation?: 'horizontal' | 'vertical'
  class?: string
}

const rootProps = withDefaults(defineProps<TabsRootProps>(), {
  orientation: 'horizontal',
})

const rootClasses = computed(() =>
  cn('flex flex-col gap-2', rootProps.class)
)
</script>

<template>
  <TabsRoot
    :default-value="defaultValue"
    :model-value="modelValue"
    :orientation="orientation"
    :class="rootClasses"
  >
    <slot />
  </TabsRoot>
</template>
