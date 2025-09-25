import MyButton from './index.vue'

export default {
  title: 'UI/Button',
  component: MyButton,
  tags: ['autodocs'],
  argTypes: {
    label: { control: 'text' },
    variant: { control: { type: 'select' }, options: ['primary', 'ghost', 'link'] },
    size: { control: { type: 'select' }, options: ['sm', 'md', 'lg'] },
    as: { control: { type: 'select' }, options: ['button', 'a', 'router-link'] },
    href: { control: 'text' },
    disabled: { control: 'boolean' },
    block: { control: 'boolean' },
    iconOnly: { control: 'boolean' },
  },
  render: (args) => ({
    components: { MyButton },
    setup() { return { args } },
    template: `<MyButton v-bind="args">{{ args.label }}</MyButton>`,
  }),
  args: {
    label: 'Button',
    variant: 'primary',
    size: 'md',
    as: 'button',
  },
}

export const Primary = {}
