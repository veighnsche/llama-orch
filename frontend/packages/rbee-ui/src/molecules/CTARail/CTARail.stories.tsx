import type { Meta, StoryObj } from '@storybook/react'
import { CTARail } from './CTARail'

const meta = {
  title: 'Molecules/CTARail',
  component: CTARail,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof CTARail>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    heading: 'Ready to get started?',
    buttons: [
      { text: 'Get Started', href: '/signup' },
      { text: 'Learn More', href: '/docs', variant: 'outline' },
    ],
  },
}

export const WithDescription: Story = {
  args: {
    heading: 'Transform Your Infrastructure',
    description: 'Join thousands of companies already using our platform',
    buttons: [
      { text: 'Start Free Trial', href: '/trial' },
      { text: 'Contact Sales', href: '/contact', variant: 'outline' },
    ],
  },
}

export const WithLinks: Story = {
  args: {
    heading: 'Need help choosing the right industry solution?',
    buttons: [
      { text: 'Schedule Consultation', href: '/consultation' },
      { text: 'View All Industries', href: '/industries', variant: 'outline' },
    ],
    links: [
      { text: 'Financial Services', href: '/industries/financial' },
      { text: 'Healthcare', href: '/industries/healthcare' },
      { text: 'Government', href: '/industries/government' },
      { text: 'Legal', href: '/industries/legal' },
    ],
  },
}

export const WithFootnote: Story = {
  args: {
    heading: 'Start Your Compliance Journey',
    description: 'Get audit-ready in weeks, not months',
    buttons: [
      { text: 'Download Compliance Pack', href: '/compliance' },
      { text: 'Talk to Expert', href: '/expert', variant: 'outline' },
    ],
    footnote: 'No credit card required • Free compliance assessment included',
  },
}

export const Complete: Story = {
  args: {
    heading: 'Need help choosing the right industry solution?',
    description: 'Our experts can guide you through the best options for your sector',
    buttons: [
      { text: 'Schedule Consultation', href: '/consultation' },
      { text: 'View All Industries', href: '/industries', variant: 'outline' },
    ],
    links: [
      { text: 'Financial Services', href: '/industries/financial' },
      { text: 'Healthcare', href: '/industries/healthcare' },
      { text: 'Government', href: '/industries/government' },
      { text: 'Legal', href: '/industries/legal' },
    ],
    footnote: 'Trusted by 500+ regulated enterprises worldwide',
  },
}

export const SingleButton: Story = {
  args: {
    heading: 'Join Our Community',
    buttons: [{ text: 'Sign Up Now', href: '/signup' }],
    links: [
      { text: 'Documentation', href: '/docs' },
      { text: 'API Reference', href: '/api' },
      { text: 'Community Forum', href: '/forum' },
    ],
  },
}
