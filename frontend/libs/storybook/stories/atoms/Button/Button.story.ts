// Created by: TEAM-FE-001
import Button from './Button.vue'

export default {
  title: 'atoms/Button',
  component: Button,
}

export const Default = () => ({
  components: { Button },
  template: '<Button>Default Button</Button>',
})

export const Destructive = () => ({
  components: { Button },
  template: '<Button variant="destructive">Destructive</Button>',
})

export const Outline = () => ({
  components: { Button },
  template: '<Button variant="outline">Outline</Button>',
})

export const Secondary = () => ({
  components: { Button },
  template: '<Button variant="secondary">Secondary</Button>',
})

export const Ghost = () => ({
  components: { Button },
  template: '<Button variant="ghost">Ghost</Button>',
})

export const Link = () => ({
  components: { Button },
  template: '<Button variant="link">Link</Button>',
})

export const Small = () => ({
  components: { Button },
  template: '<Button size="sm">Small</Button>',
})

export const Large = () => ({
  components: { Button },
  template: '<Button size="lg">Large</Button>',
})

export const Icon = () => ({
  components: { Button },
  template: '<Button size="icon">🔥</Button>',
})

export const IconSmall = () => ({
  components: { Button },
  template: '<Button size="icon-sm">🔥</Button>',
})

export const IconLarge = () => ({
  components: { Button },
  template: '<Button size="icon-lg">🔥</Button>',
})

export const Disabled = () => ({
  components: { Button },
  template: '<Button disabled>Disabled</Button>',
})

export const AllVariants = () => ({
  components: { Button },
  template: `
    <div style="display: flex; flex-direction: column; gap: 16px;">
      <div style="display: flex; gap: 8px; flex-wrap: wrap;">
        <Button>Default</Button>
        <Button variant="destructive">Destructive</Button>
        <Button variant="outline">Outline</Button>
        <Button variant="secondary">Secondary</Button>
        <Button variant="ghost">Ghost</Button>
        <Button variant="link">Link</Button>
      </div>
      <div style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center;">
        <Button size="sm">Small</Button>
        <Button>Default</Button>
        <Button size="lg">Large</Button>
      </div>
      <div style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center;">
        <Button size="icon-sm">🔥</Button>
        <Button size="icon">🔥</Button>
        <Button size="icon-lg">🔥</Button>
      </div>
      <div style="display: flex; gap: 8px; flex-wrap: wrap;">
        <Button disabled>Disabled</Button>
        <Button variant="outline" disabled>Disabled Outline</Button>
      </div>
    </div>
  `,
})
