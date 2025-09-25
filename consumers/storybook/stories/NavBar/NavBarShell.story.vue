<script setup lang="ts">
import { reactive, computed, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import NavBarShell from './NavBarShell.vue'
import Brand from './Brand/Brand.vue'
import NavLinks from './NavLinks.vue'
import Button from './Button/Button.vue'
import Drawer from './Drawer.vue'
import DrawerTrigger from './DrawerTrigger.vue'
import DrawerPanel from './DrawerPanel.vue'

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
watch(
  () => route.fullPath,
  () => (open.value = false),
)
</script>

<template>
  <Story title="UI/NavBarShell" :layout="{ type: 'single', iframe: false }">
    <Variant title="Playground">
      <Drawer v-model="open" :hide-on-desktop="true">
        <NavBarShell :nav-aria-label="state.navLabel">
          <template #brand>
            <Brand :brand="brand" />
          </template>
          <template #toggle>
            <DrawerTrigger :as="Button" variant="ghost" size="sm" iconOnly>
              <svg v-if="!open" viewBox="0 0 24 24" aria-hidden="true">
                <path
                  d="M4 7h16M4 12h16M4 17h16"
                  stroke="currentColor"
                  stroke-width="2"
                  stroke-linecap="round"
                />
              </svg>
              <svg v-else viewBox="0 0 24 24" aria-hidden="true">
                <path
                  d="M6 6l12 12M18 6l-12 12"
                  stroke="currentColor"
                  stroke-width="2"
                  stroke-linecap="round"
                />
              </svg>
            </DrawerTrigger>
          </template>
          <template #links>
            <NavLinks :items="state.items" />
          </template>
          <template #right>
            <Button as="router-link" :to="'/contact'" size="sm" variant="primary"
              >Contact us</Button
            >
          </template>
          <template #drawer>
            <DrawerPanel :items="state.items">
              <template #ops>
                <Button as="router-link" :to="'/contact'" variant="primary">Contact us</Button>
              </template>
            </DrawerPanel>
          </template>
        </NavBarShell>
      </Drawer>
    </Variant>

    <template #controls>
      <HstText v-model="state.brandLabel" title="Brand label" />
      <HstCheckbox v-model="state.showGlyph" title="Show glyph" />
      <HstText v-model="state.navLabel" title="ARIA label" />
      <HstJson v-model="state.items" title="Links (array of {label, to|href})" />
    </template>
  </Story>
</template>
