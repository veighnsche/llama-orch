<script setup lang="ts">
  import { reactive, computed } from 'vue'
  import Brand from './Brand.vue'

  const state = reactive({
    label: 'Orchyra',
    showGlyph: true,
    to: '/',
    href: '',
  })

  const brand = computed(() => ({
    label: state.label,
    to: state.href ? undefined : state.to || '/',
    href: state.href || undefined,
    ariaLabel: `${state.label} — home`,
    showGlyph: state.showGlyph,
  }))
</script>

<template>
  <Story title="UI/Brand" :layout="{ type: 'single', iframe: false }">
    <Variant title="Playground">
      <Brand :brand="brand" />
    </Variant>

    <Variant title="Router">
      <Brand :brand="{ label: 'Orchyra', to: '/' }" />
    </Variant>

    <Variant title="External">
      <Brand :brand="{ label: 'Orchyra', href: 'https://example.com', showGlyph: false }" />
    </Variant>

    <template #controls>
      <HstText v-model="state.label" title="label" />
      <HstCheckbox v-model="state.showGlyph" title="showGlyph" />
      <HstSelect v-model="state.to" title="to (router)" :options="['/', '/about', '/contact']" />
      <HstText v-model="state.href" title="href (external)" />
    </template>
  </Story>
</template>
