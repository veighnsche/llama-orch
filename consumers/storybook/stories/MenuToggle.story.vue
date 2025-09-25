<script setup lang="ts">
import { reactive, ref } from 'vue'
import MenuToggle from './MenuToggle.vue'

const labels = reactive({ open: 'Open menu', close: 'Close menu', srOnly: 'Menu' })
const open = ref(false)
const size = ref<'sm' | 'md' | 'lg'>('md')
const panelId = 'menu-toggle-panel'
</script>

<template>
  <Story title="UI/MenuToggle" :layout="{ type: 'single', iframe: false }">
    <Variant title="Playground">
      <div style="display: flex; gap: 12px; align-items: center">
        <MenuToggle v-model:open="open" :labels="labels" :size="size" :aria-controls="panelId">
          <template #icon-closed>
            <!-- Custom closed icon -->
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 6h18M3 12h18M3 18h18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
          </template>
          <template #icon-open>
            <!-- Custom open icon -->
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M6 18L18 6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
          </template>
        </MenuToggle>
        <span>Open: {{ open ? 'yes' : 'no' }}</span>
      </div>
      <div :id="panelId" v-show="open" style="margin-top: 8px; padding: 8px; border: 1px solid var(--surface-muted); border-radius: var(--radius-md);">
        Controlled content panel (toggles with the button)
      </div>
    </Variant>

    <template #controls>
      <HstCheckbox v-model="open" title="open" />
      <HstSelect v-model="size" title="size" :options="['sm','md','lg']" />
      <HstText v-model="labels.open" title="labels.open" />
      <HstText v-model="labels.close" title="labels.close" />
      <HstText v-model="labels.srOnly" title="labels.srOnly" />
    </template>
  </Story>
</template>
