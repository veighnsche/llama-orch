import { technicalTemplateProps } from '@rbee/ui/pages/HomePage'
import type { Meta, StoryObj } from '@storybook/react'
import { TechnicalTemplate } from './TechnicalTemplate'

const meta = {
  title: 'Templates/TechnicalTemplate',
  component: TechnicalTemplate,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof TechnicalTemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * TechnicalTemplate as used on the Home page
 * - Five architecture highlights (BDD, Cascading Shutdown, Process Isolation, etc.)
 * - BDD coverage progress (42/62)
 * - RbeeArch diagram
 * - Five tech stack items (Rust, Candle ML, Rhai, SQLite, Axum + Vue.js)
 * - GitHub and architecture links
 */
export const OnHomePage: Story = {
  args: technicalTemplateProps,
}
