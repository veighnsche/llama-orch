import type { Meta, StoryObj } from '@storybook/react'
import { CTABanner } from './CTABanner'

const meta: Meta<typeof CTABanner> = {
  title: 'Organisms/Footers/CTABanner',
  component: CTABanner,
  parameters: {
    layout: 'centered',
  },
}

export default meta
type Story = StoryObj<typeof CTABanner>

export const WithBothButtons: Story = {
  args: {
    copy: 'Ready to get started with rbee?',
    primary: {
      label: 'Get Started Free',
      href: '/getting-started',
    },
    secondary: {
      label: 'View Documentation',
      href: '/docs',
    },
  },
}

export const PrimaryOnly: Story = {
  args: {
    copy: 'Start building with AI infrastructure on your terms',
    primary: {
      label: 'Sign Up Now',
      href: '/signup',
    },
  },
}

export const SecondaryOnly: Story = {
  args: {
    copy: 'Want to learn more about rbee?',
    secondary: {
      label: 'Read the Docs',
      href: '/docs',
    },
  },
}

export const ButtonsOnly: Story = {
  args: {
    primary: {
      label: 'Get Started',
      href: '/start',
    },
    secondary: {
      label: 'Contact Sales',
      href: '/contact',
    },
  },
}

export const LongCopy: Story = {
  args: {
    copy: 'Join thousands of developers who are already building AI applications with rbee. Get started in minutes with our comprehensive documentation and active community support.',
    primary: {
      label: 'Get Started Free',
      href: '/start',
      ariaLabel: 'Get started with rbee for free',
    },
    secondary: {
      label: 'Talk to Sales',
      href: '/sales',
    },
  },
}

export const ShortCopy: Story = {
  args: {
    copy: 'Questions?',
    primary: {
      label: 'Contact Us',
      href: '/contact',
    },
  },
}
