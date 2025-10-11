// Created by: TEAM-FE-000 (Scaffolding)
// TEAM-FE-001: Implemented Switch story with all variants

import { ref } from 'vue'
import Switch from './Switch.vue'

export default {
  title: 'atoms/Switch',
  component: Switch,
}

export const Default = () => ({
  components: { Switch },
  setup() {
    const checked = ref(false)
    return { checked }
  },
  template: '<Switch v-model:checked="checked" />',
})

export const Checked = () => ({
  components: { Switch },
  setup() {
    const checked = ref(true)
    return { checked }
  },
  template: '<Switch v-model:checked="checked" />',
})

export const Disabled = () => ({
  components: { Switch },
  template: '<Switch disabled />',
})

export const DisabledChecked = () => ({
  components: { Switch },
  template: '<Switch disabled :checked="true" />',
})

export const WithLabel = () => ({
  components: { Switch },
  setup() {
    const checked = ref(false)
    return { checked }
  },
  template: `
    <div style="display: flex; align-items: center; gap: 8px;">
      <Switch v-model:checked="checked" id="airplane-mode" />
      <label for="airplane-mode" style="cursor: pointer;">Airplane Mode</label>
    </div>
  `,
})

export const AllStates = () => ({
  components: { Switch },
  setup() {
    const checked1 = ref(false)
    const checked2 = ref(true)
    return { checked1, checked2 }
  },
  template: `
    <div style="display: flex; flex-direction: column; gap: 16px;">
      <div style="display: flex; align-items: center; gap: 8px;">
        <Switch v-model:checked="checked1" id="off" />
        <label for="off">Off</label>
      </div>
      <div style="display: flex; align-items: center; gap: 8px;">
        <Switch v-model:checked="checked2" id="on" />
        <label for="on">On</label>
      </div>
      <div style="display: flex; align-items: center; gap: 8px;">
        <Switch disabled id="disabled-off" />
        <label for="disabled-off">Disabled Off</label>
      </div>
      <div style="display: flex; align-items: center; gap: 8px;">
        <Switch disabled :checked="true" id="disabled-on" />
        <label for="disabled-on">Disabled On</label>
      </div>
    </div>
  `,
})
