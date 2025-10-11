// Created by: TEAM-FE-000 (Scaffolding)
// TEAM-FE-001: Implemented RadioGroup story with all variants

import { ref } from 'vue'
import RadioGroup from './RadioGroup.vue'
import RadioGroupItem from './RadioGroupItem.vue'

export default {
  title: 'atoms/RadioGroup',
  component: RadioGroup,
}

export const Default = () => ({
  components: { RadioGroup, RadioGroupItem },
  setup() {
    const selected = ref('option1')
    return { selected }
  },
  template: `
    <RadioGroup v-model="selected">
      <div style="display: flex; align-items: center; gap: 8px;">
        <RadioGroupItem value="option1" id="option1" />
        <label for="option1">Option 1</label>
      </div>
      <div style="display: flex; align-items: center; gap: 8px;">
        <RadioGroupItem value="option2" id="option2" />
        <label for="option2">Option 2</label>
      </div>
      <div style="display: flex; align-items: center; gap: 8px;">
        <RadioGroupItem value="option3" id="option3" />
        <label for="option3">Option 3</label>
      </div>
    </RadioGroup>
  `,
})

export const WithDisabled = () => ({
  components: { RadioGroup, RadioGroupItem },
  setup() {
    const selected = ref('comfortable')
    return { selected }
  },
  template: `
    <RadioGroup v-model="selected">
      <div style="display: flex; align-items: center; gap: 8px;">
        <RadioGroupItem value="default" id="default" />
        <label for="default">Default</label>
      </div>
      <div style="display: flex; align-items: center; gap: 8px;">
        <RadioGroupItem value="comfortable" id="comfortable" />
        <label for="comfortable">Comfortable</label>
      </div>
      <div style="display: flex; align-items: center; gap: 8px;">
        <RadioGroupItem value="compact" id="compact" disabled />
        <label for="compact">Compact (Disabled)</label>
      </div>
    </RadioGroup>
  `,
})

export const AllStates = () => ({
  components: { RadioGroup, RadioGroupItem },
  setup() {
    const selected = ref('card')
    return { selected }
  },
  template: `
    <div style="display: flex; flex-direction: column; gap: 24px;">
      <div>
        <h3 style="margin-bottom: 12px; font-weight: 600;">Payment Method</h3>
        <RadioGroup v-model="selected">
          <div style="display: flex; align-items: center; gap: 8px;">
            <RadioGroupItem value="card" id="card" />
            <label for="card" style="cursor: pointer;">Credit Card</label>
          </div>
          <div style="display: flex; align-items: center; gap: 8px;">
            <RadioGroupItem value="paypal" id="paypal" />
            <label for="paypal" style="cursor: pointer;">PayPal</label>
          </div>
          <div style="display: flex; align-items: center; gap: 8px;">
            <RadioGroupItem value="apple" id="apple" />
            <label for="apple" style="cursor: pointer;">Apple Pay</label>
          </div>
        </RadioGroup>
      </div>
      <div>
        <p style="margin-top: 8px; font-size: 14px; color: #666;">Selected: {{ selected }}</p>
      </div>
    </div>
  `,
})
