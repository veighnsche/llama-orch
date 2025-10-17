import type { Meta, StoryObj } from '@storybook/react'
import { SectionCTAs } from './SectionCTAs'

const meta: Meta<typeof SectionCTAs> = {
  title: 'Organisms/Footers/SectionCTAs',
  component: SectionCTAs,
  parameters: {
    layout: 'centered',
  },
}

export default meta
type Story = StoryObj<typeof SectionCTAs>

export const Complete: Story = {
  args: {
    label: 'Ready to get started?',
    primary: {
      label: 'Get Started Free',
      href: '/getting-started',
    },
    secondary: {
      label: 'View Documentation',
      href: '/docs',
    },
    caption: 'No credit card required • Cancel anytime',
  },
}

export const WithBothButtons: Story = {
  args: {
    primary: {
      label: 'Sign Up Now',
      href: '/signup',
    },
    secondary: {
      label: 'Contact Sales',
      href: '/contact',
    },
  },
}

export const PrimaryOnly: Story = {
  args: {
    label: 'Start building with rbee today',
    primary: {
      label: 'Get Started',
      href: '/start',
    },
    caption: 'Free forever for personal use',
  },
}

export const SecondaryOnly: Story = {
  args: {
    label: 'Want to learn more?',
    secondary: {
      label: 'Read the Docs',
      href: '/docs',
    },
  },
}

export const WithLabel: Story = {
  args: {
    label: "Questions? We're here to help",
    primary: {
      label: 'Contact Support',
      href: '/support',
    },
  },
}

export const WithCaption: Story = {
  args: {
    primary: {
      label: 'Try rbee Free',
      href: '/trial',
    },
    caption: '14-day trial • No credit card required',
  },
}

export const LabelAndCaption: Story = {
  args: {
    label: 'Join thousands of developers',
    primary: {
      label: 'Get Started',
      href: '/start',
    },
    secondary: {
      label: 'View Pricing',
      href: '/pricing',
    },
    caption: 'Trusted by companies worldwide',
  },
}

export const MinimalPrimary: Story = {
  args: {
    primary: {
      label: 'Get Started',
      href: '/start',
    },
  },
}

export const MinimalSecondary: Story = {
  args: {
    secondary: {
      label: 'Learn More',
      href: '/learn',
    },
  },
}

export const LongCaption: Story = {
  args: {
    label: 'Ready to revolutionize your AI infrastructure?',
    primary: {
      label: 'Start Your Free Trial',
      href: '/trial',
      ariaLabel: 'Start your 14-day free trial of rbee',
    },
    secondary: {
      label: 'Schedule a Demo',
      href: '/demo',
    },
    caption:
      'Join over 10,000 developers building AI applications with rbee. Free forever for personal projects, flexible pricing for teams.',
  },
}
