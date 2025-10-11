// Created by: TEAM-FE-000 (Scaffolding)
// TEAM-FE-001: Implemented Checkbox story with all variants

import { ref } from 'vue'
import Checkbox from './Checkbox.vue'

export default {
  title: 'atoms/Checkbox',
  component: Checkbox,
}

export const Default = () => ({
  components: { Checkbox },
  setup() {
    const checked = ref(false)
    return { checked }
  },
  template: '<Checkbox v-model:checked="checked" />',
})

export const Checked = () => ({
  components: { Checkbox },
  setup() {
    const checked = ref(true)
    return { checked }
  },
  template: '<Checkbox v-model:checked="checked" />',
})

export const Disabled = () => ({
  components: { Checkbox },
  template: '<Checkbox disabled />',
})

export const DisabledChecked = () => ({
  components: { Checkbox },
  template: '<Checkbox disabled :checked="true" />',
})

export const WithLabel = () => ({
  components: { Checkbox },
  setup() {
    const checked = ref(false)
    return { checked }
  },
  template: `
    <div style="display: flex; align-items: center; gap: 8px;">
      <Checkbox v-model:checked="checked" id="terms" />
      <label for="terms" style="cursor: pointer;">Accept terms and conditions</label>
    </div>
  `,
})

export const AllStates = () => ({
  components: { Checkbox },
  setup() {
    const checked1 = ref(false)
    const checked2 = ref(true)
    return { checked1, checked2 }
  },
  template: `
    <div style="display: flex; flex-direction: column; gap: 16px;">
      <div style="display: flex; align-items: center; gap: 8px;">
        <Checkbox v-model:checked="checked1" id="unchecked" />
        <label for="unchecked">Unchecked</label>
      </div>
      <div style="display: flex; align-items: center; gap: 8px;">
        <Checkbox v-model:checked="checked2" id="checked" />
        <label for="checked">Checked</label>
      </div>
      <div style="display: flex; align-items: center; gap: 8px;">
        <Checkbox disabled id="disabled" />
        <label for="disabled">Disabled</label>
      </div>
      <div style="display: flex; align-items: center; gap: 8px;">
        <Checkbox disabled :checked="true" id="disabled-checked" />
        <label for="disabled-checked">Disabled Checked</label>
      </div>
    </div>
  `,
})
