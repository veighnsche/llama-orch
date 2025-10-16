import { homeHeroProps } from '@rbee/ui/pages/HomePage/HomePageProps'
import type { Meta, StoryObj } from '@storybook/react'
import { HomeHero } from './HomeHero'

const meta = {
  title: 'Templates/HomeHero',
  component: HomeHero,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof HomeHero>

export default meta
type Story = StoryObj<typeof meta>

export const OnHomePage: Story = {
  args: homeHeroProps,
}
