<script setup lang="ts">
  import { inject, useAttrs } from 'vue'
  import type { DrawerContext } from './Drawer.vue'

  const props = withDefaults(
    defineProps<{
      as?: string | object
      ariaControls?: string
    }>(),
    {
      as: 'button',
      ariaControls: undefined,
    },
  )

  const attrs = useAttrs()
  const ctx = inject<DrawerContext>('drawer-ctx')

  function onClick(e: MouseEvent) {
    e.stopPropagation()
    ctx?.toggle()
  }
</script>

<template>
  <component
    :is="props.as"
    type="button"
    :aria-expanded="ctx?.open ? 'true' : 'false'"
    :aria-controls="props.ariaControls || ctx?.id"
    v-bind="attrs"
    @click="onClick"
  >
    <slot />
  </component>
</template>
