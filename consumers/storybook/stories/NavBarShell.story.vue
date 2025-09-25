<script setup lang="ts">
import { reactive, computed, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import NavBarShell from './NavBarShell.vue'
import Brand from './Brand.vue'
import MenuToggle from './MenuToggle.vue'
import NavLinks from './NavLinks.vue'
import Drawer from './Drawer.vue'
import Button from './Button.vue'

const state = reactive({
  brandLabel: 'Orchyra',
  showGlyph: true,
  navLabel: 'Primary navigation',
  items: [
    { label: 'Home', to: '/' },
    { label: 'About', to: '/about' },
    { label: 'Contact', to: '/contact' },
  ] as Array<{ label: string; to?: string; href?: string }>,
})

const brand = computed(() => ({
  label: state.brandLabel,
  to: '/',
  ariaLabel: `${state.brandLabel} — home`,
  showGlyph: state.showGlyph,
}))

const open = ref(false)
const route = useRoute()
watch(() => route.fullPath, () => (open.value = false))
</script>

<template>
  <Story title="UI/NavBarShell" :layout="{ type: 'single', iframe: false }">
    <Variant title="Playground">
      <NavBarShell :nav-aria-label="state.navLabel">
        <template #brand>
          <Brand :brand="brand" />
        </template>
        <template #toggle>
          <MenuToggle :open="open" @toggle="open = !open" />
        </template>
        <template #links>
          <NavLinks :items="state.items" />
        </template>
        <template #right>
          <Button as="router-link" :to="'/contact'" size="sm" variant="primary">Contact us</Button>
        </template>
        <template #drawer>
          <Drawer :open="open" :items="state.items" @close="open = false">
            <template #ops>
              <Button as="router-link" :to="'/contact'" variant="primary">Contact us</Button>
            </template>
          </Drawer>
        </template>
      </NavBarShell>
    </Variant>

    <template #controls>
      <HstText v-model="state.brandLabel" title="Brand label" />
      <HstCheckbox v-model="state.showGlyph" title="Show glyph" />
      <HstText v-model="state.navLabel" title="ARIA label" />
      <HstJson v-model="state.items" title="Links (array of {label, to|href})" />
    </template>
  </Story>
</template>
