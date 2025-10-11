// Created by: TEAM-FE-000 (Scaffolding)
// TEAM-FE-001: Implemented Slider story with all variants

import { ref } from 'vue'
import Slider from './Slider.vue'

export default {
  title: 'atoms/Slider',
  component: Slider,
}

export const Default = () => ({
  components: { Slider },
  setup() {
    const value = ref([50])
    return { value }
  },
  template: `
    <div style="width: 300px;">
      <Slider v-model="value" />
      <p style="margin-top: 12px;">Value: {{ value[0] }}</p>
    </div>
  `,
})

export const WithRange = () => ({
  components: { Slider },
  setup() {
    const value = ref([25, 75])
    return { value }
  },
  template: `
    <div style="width: 300px;">
      <Slider v-model="value" />
      <p style="margin-top: 12px;">Range: {{ value[0] }} - {{ value[1] }}</p>
    </div>
  `,
})

export const WithStep = () => ({
  components: { Slider },
  setup() {
    const value = ref([50])
    return { value }
  },
  template: `
    <div style="width: 300px;">
      <Slider v-model="value" :step="10" />
      <p style="margin-top: 12px;">Value (step 10): {{ value[0] }}</p>
    </div>
  `,
})

export const WithMinMax = () => ({
  components: { Slider },
  setup() {
    const value = ref([20])
    return { value }
  },
  template: `
    <div style="width: 300px;">
      <Slider v-model="value" :min="0" :max="50" />
      <p style="margin-top: 12px;">Value (0-50): {{ value[0] }}</p>
    </div>
  `,
})

export const Disabled = () => ({
  components: { Slider },
  template: `
    <div style="width: 300px;">
      <Slider :default-value="[50]" disabled />
    </div>
  `,
})

export const AllVariants = () => ({
  components: { Slider },
  setup() {
    const value1 = ref([50])
    const value2 = ref([25, 75])
    const value3 = ref([30])
    return { value1, value2, value3 }
  },
  template: `
    <div style="display: flex; flex-direction: column; gap: 32px; width: 400px;">
      <div>
        <label style="display: block; margin-bottom: 8px; font-weight: 500;">Single Value</label>
        <Slider v-model="value1" />
        <p style="margin-top: 8px; font-size: 14px; color: #666;">Value: {{ value1[0] }}</p>
      </div>
      
      <div>
        <label style="display: block; margin-bottom: 8px; font-weight: 500;">Range</label>
        <Slider v-model="value2" />
        <p style="margin-top: 8px; font-size: 14px; color: #666;">Range: {{ value2[0] }} - {{ value2[1] }}</p>
      </div>
      
      <div>
        <label style="display: block; margin-bottom: 8px; font-weight: 500;">With Step (10)</label>
        <Slider v-model="value3" :step="10" />
        <p style="margin-top: 8px; font-size: 14px; color: #666;">Value: {{ value3[0] }}</p>
      </div>
      
      <div>
        <label style="display: block; margin-bottom: 8px; font-weight: 500;">Disabled</label>
        <Slider :default-value="[60]" disabled />
      </div>
    </div>
  `,
})
