<script setup lang="ts">
import { reactive } from 'vue'
import NavLinks from './NavLinks.vue'

const state = reactive({
  items: [
    { label: 'Home', to: '/' },
    { label: 'About', to: '/about' },
    { label: 'Contact', to: '/contact' },
  ] as Array<{ label: string; to?: string; href?: string }>,
})
</script>

<template>
  <Story title="UI/NavLinks" :layout="{ type: 'grid', width: 980 }">
    <Variant title="Playground">
      <!-- Set a wide width so desktop styles apply -->
      <NavLinks :items="state.items" />
    </Variant>

    <Variant title="With external">
      <NavLinks :items="[{ label: 'Docs', href: 'https://example.com' }, { label: 'About', to: '/about' }]" />
    </Variant>

    <template #controls>
      <HstJson v-model="state.items" title="items (array of {label,to|href})" />
    </template>
  </Story>
</template>
