// Created by: TEAM-FE-000 (Scaffolding)
// TEAM-FE-001: Implemented Textarea story with all variants

import Textarea from './Textarea.vue'

export default {
  title: 'atoms/Textarea',
  component: Textarea,
}

export const Default = () => ({
  components: { Textarea },
  template: '<Textarea placeholder="Enter text..." />',
})

export const WithRows = () => ({
  components: { Textarea },
  template: '<Textarea :rows="5" placeholder="Larger textarea with 5 rows..." />',
})

export const Disabled = () => ({
  components: { Textarea },
  template: '<Textarea disabled placeholder="Disabled textarea" />',
})

export const Readonly = () => ({
  components: { Textarea },
  template: '<Textarea readonly value="This is read-only text that cannot be edited." />',
})

export const AllVariants = () => ({
  components: { Textarea },
  template: `
    <div style="display: flex; flex-direction: column; gap: 16px; max-width: 500px;">
      <Textarea placeholder="Default textarea" />
      <Textarea :rows="5" placeholder="Textarea with 5 rows" />
      <Textarea disabled placeholder="Disabled textarea" />
      <Textarea readonly value="Read-only textarea content" />
    </div>
  `,
})
