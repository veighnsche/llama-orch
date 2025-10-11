// Created by: TEAM-FE-000 (Scaffolding)
// TEAM-FE-001: Implemented Input story with all variants

import Input from './Input.vue'

export default {
  title: 'atoms/Input',
  component: Input,
}

export const Default = () => ({
  components: { Input },
  template: '<Input placeholder="Enter text..." />',
})

export const Email = () => ({
  components: { Input },
  template: '<Input type="email" placeholder="Enter email..." />',
})

export const Password = () => ({
  components: { Input },
  template: '<Input type="password" placeholder="Enter password..." />',
})

export const Number = () => ({
  components: { Input },
  template: '<Input type="number" placeholder="Enter number..." />',
})

export const Search = () => ({
  components: { Input },
  template: '<Input type="search" placeholder="Search..." />',
})

export const Tel = () => ({
  components: { Input },
  template: '<Input type="tel" placeholder="Enter phone..." />',
})

export const Url = () => ({
  components: { Input },
  template: '<Input type="url" placeholder="Enter URL..." />',
})

export const Disabled = () => ({
  components: { Input },
  template: '<Input disabled placeholder="Disabled input" />',
})

export const Readonly = () => ({
  components: { Input },
  template: '<Input readonly value="Read-only value" />',
})

export const AllTypes = () => ({
  components: { Input },
  template: `
    <div style="display: flex; flex-direction: column; gap: 12px; max-width: 400px;">
      <Input placeholder="Text input" />
      <Input type="email" placeholder="Email input" />
      <Input type="password" placeholder="Password input" />
      <Input type="number" placeholder="Number input" />
      <Input type="search" placeholder="Search input" />
      <Input type="tel" placeholder="Phone input" />
      <Input type="url" placeholder="URL input" />
      <Input disabled placeholder="Disabled input" />
      <Input readonly value="Read-only input" />
    </div>
  `,
})
