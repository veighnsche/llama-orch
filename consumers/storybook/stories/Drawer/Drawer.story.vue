<script setup lang="ts">
  import { reactive, ref } from 'vue'
  import Drawer from './Drawer.vue'
  import DrawerPanel from './DrawerPanel.vue'
  import DrawerTrigger from './DrawerTrigger.vue'
  import Button from '../Button/Button.vue'

  const state = reactive({
    items: [
      { label: 'Home', to: '/' },
      { label: 'About', to: '/about' },
      { label: 'Contact', to: '/contact' },
    ] as Array<{ label: string; to?: string; href?: string }>,
  })

  // States per variant
  const openItems = ref(false)
  const openCustom = ref(false)

  // A11y + behavior controls
  const hideOnDesktop = ref(false)
  const role = ref<'dialog' | 'menu' | 'navigation'>('dialog')
  const ariaModal = ref(true)
  const label = ref('Navigation')
  const labelledBy = ref('drawer-title')
  const drawerId = 'drawer-story'
</script>

<template>
  <Story title="Behavior/Drawer" :layout="{ type: 'single', width: 360 }">
    <Variant title="Items (v-model)">
      <div style="padding: 8px; display: flex; gap: 8px; align-items: center">
        <Button variant="ghost" @click="openItems = true"> Open drawer </Button>
        <span>Open: {{ openItems ? 'yes' : 'no' }}</span>
      </div>
      <Drawer
        :id="drawerId"
        v-model="openItems"
        :hide-on-desktop="hideOnDesktop"
        :aria-role="role"
        :aria-modal="ariaModal"
        :label="label"
        :labelled-by="labelledBy"
      >
        <DrawerPanel :items="state.items">
          <h3 v-if="labelledBy" :id="labelledBy" style="margin: 0 0 8px 0">
            {{ label }}
          </h3>
          <template #ops>
            <Button as="router-link" :to="'/contact'" variant="primary"> Contact us </Button>
          </template>
        </DrawerPanel>
      </Drawer>
    </Variant>

    <Variant title="Custom content">
      <div style="padding: 8px; display: flex; gap: 8px; align-items: center">
        <Button variant="ghost" @click="openCustom = true"> Open custom drawer </Button>
        <!-- Example internal trigger using DrawerTrigger -->
        <Drawer
          v-model="openCustom"
          :hide-on-desktop="hideOnDesktop"
          :aria-role="role"
          :aria-modal="ariaModal"
        >
          <div style="padding: 8px; display: flex; gap: 8px; align-items: center">
            <DrawerTrigger :as="Button" variant="ghost" size="sm"> Toggle </DrawerTrigger>
          </div>
          <DrawerPanel>
            <div style="padding: 6px 2px">
              <p style="margin: 0 0 8px 0">
                This drawer shows arbitrary content via the default slot.
              </p>
              <ul style="margin: 0; padding-left: 18px">
                <li>Works with ESC to close</li>
                <li>Locks scroll while open</li>
                <li>Restores focus on close</li>
              </ul>
            </div>
            <template #ops>
              <Button variant="primary" @click="openCustom = false"> Close </Button>
            </template>
          </DrawerPanel>
        </Drawer>
      </div>
    </Variant>

    <template #controls>
      <HstCheckbox v-model="openItems" title="open (items)" />
      <HstCheckbox v-model="openCustom" title="open (custom)" />
      <HstCheckbox v-model="hideOnDesktop" title="hideOnDesktop" />
      <HstSelect v-model="role" title="ariaRole" :options="['dialog', 'menu', 'navigation']" />
      <HstCheckbox v-model="ariaModal" title="ariaModal" />
      <HstText v-model="label" title="label" />
      <HstText v-model="labelledBy" title="labelledBy (element id)" />
      <HstJson v-model="state.items" title="items (array of {label,to|href})" />
      <p>Tip: Keep preview width below 920px to see the drawer when hideOnDesktop is true.</p>
    </template>
  </Story>
</template>
