import { whatIsRbeeProps } from '@rbee/ui/pages/HomePage/HomePageProps'
import type { Meta, StoryObj } from '@storybook/react'
import { WhatIsRbee } from './WhatIsRbee'

const meta = {
  title: 'Templates/WhatIsRbee',
  component: WhatIsRbee,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof WhatIsRbee>

export default meta
type Story = StoryObj<typeof meta>

export const OnHomePage: Story = {
  args: whatIsRbeeProps,
}
