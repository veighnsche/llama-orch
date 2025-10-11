// Created by: TEAM-FE-000 (Scaffolding)
// TEAM-FE-001: Implemented Label story with all variants

import Label from './Label.vue'

export default {
  title: 'atoms/Label',
  component: Label,
}

export const Default = () => ({
  components: { Label },
  template: '<Label>Default Label</Label>',
})

export const Required = () => ({
  components: { Label },
  template: '<Label required>Required Label</Label>',
})

export const Disabled = () => ({
  components: { Label },
  template: '<Label disabled>Disabled Label</Label>',
})

export const WithFor = () => ({
  components: { Label },
  template: '<Label for="input-id">Label for Input</Label>',
})

export const AllStates = () => ({
  components: { Label },
  template: `
    <div style="display: flex; flex-direction: column; gap: 12px;">
      <Label>Default Label</Label>
      <Label required>Required Label</Label>
      <Label disabled>Disabled Label</Label>
      <Label for="example-input">Label with for attribute</Label>
    </div>
  `,
})
