<script setup lang="ts">
import { ref, reactive } from 'vue'
import DrawerRoot from './DrawerRoot.vue'
import DrawerTrigger from './DrawerTrigger.vue'
import DrawerPanel from './DrawerPanel.vue'
import Button from './Button.vue'
import MenuToggle from './MenuToggle.vue'

const state = reactive({
  items: [
    { label: 'Home', to: '/' },
    { label: 'About', to: '/about' },
    { label: 'Contact', to: '/contact' },
  ] as Array<{ label: string; to?: string; href?: string }>,
})

const open = ref(false)
const hideOnDesktop = ref(false)
const role = ref<'dialog' | 'menu' | 'navigation'>('dialog')
const ariaModal = ref(true)
const label = ref('Navigation')
const labelledBy = ref('drawer-root-title')
const id = 'drawer-root'
</script>

<template>
  <Story title="UI/DrawerRoot (compound)" :layout="{ type: 'single', width: 420 }">
    <Variant title="Button trigger">
      <DrawerRoot v-model="open" :id="id" :hide-on-desktop="hideOnDesktop" :aria-role="role" :aria-modal="ariaModal" :label="label" :labelled-by="labelledBy">
        <div style="display: flex; align-items: center; gap: 8px; padding: 8px">
          <DrawerTrigger :as="Button">Open</DrawerTrigger>
          <span>Open: {{ open ? 'yes' : 'no' }}</span>
        </div>
        <DrawerPanel :items="state.items">
          <h3 :id="labelledBy" style="margin: 0 0 8px 0">{{ label }}</h3>
          <template #ops>
            <Button variant="primary" @click="open = false">Close</Button>
          </template>
        </DrawerPanel>
      </DrawerRoot>
    </Variant>

    <Variant title="MenuToggle trigger + custom content">
      <DrawerRoot v-model="open" :id="id + '-2'" :hide-on-desktop="hideOnDesktop" :aria-role="role" :aria-modal="ariaModal">
        <div style="display: flex; align-items: center; gap: 8px; padding: 8px">
          <DrawerTrigger :as="MenuToggle" />
        </div>
        <DrawerPanel>
          <div style="padding: 6px 2px">
            <p style="margin: 0 0 8px 0">This drawer shows custom content via the default slot.</p>
            <ul style="margin: 0; padding-left: 18px">
              <li>ESC to close</li>
              <li>Click overlay to close</li>
              <li>Focus restored on close</li>
            </ul>
          </div>
          <template #ops>
            <Button variant="primary" @click="open = false">Close</Button>
          </template>
        </DrawerPanel>
      </DrawerRoot>
    </Variant>

    <template #controls>
      <HstCheckbox v-model="open" title="open" />
      <HstCheckbox v-model="hideOnDesktop" title="hideOnDesktop" />
      <HstSelect v-model="role" title="ariaRole" :options="['dialog','menu','navigation']" />
      <HstCheckbox v-model="ariaModal" title="ariaModal" />
      <HstText v-model="label" title="label" />
      <HstText v-model="labelledBy" title="labelledBy (element id)" />
      <HstJson v-model="state.items" title="items (array of {label,to|href})" />
    </template>
  </Story>
</template>
