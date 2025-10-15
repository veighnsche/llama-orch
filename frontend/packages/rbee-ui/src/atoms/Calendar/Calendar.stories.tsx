// Created by: TEAM-011
import type { Meta, StoryObj } from '@storybook/react'
import { useState } from 'react'
import { Calendar } from './Calendar'

const meta: Meta<typeof Calendar> = {
  title: 'Atoms/Calendar',
  component: Calendar,
  parameters: { layout: 'centered' },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof Calendar>

export const Default: Story = {
  render: () => {
    const [date, setDate] = useState<Date | undefined>(new Date())
    return <Calendar mode="single" selected={date} onSelect={setDate} />
  },
}

export const WithRange: Story = {
  render: () => {
    const [range, setRange] = useState<{ from: Date | undefined; to: Date | undefined }>({
      from: new Date(),
      to: undefined,
    })
    return <Calendar mode="range" selected={range} onSelect={setRange as any} />
  },
}

export const WithDisabled: Story = {
  render: () => {
    const [date, setDate] = useState<Date | undefined>(new Date())
    return (
      <Calendar
        mode="single"
        selected={date}
        onSelect={setDate}
        disabled={(date) => date < new Date() || date > new Date(Date.now() + 30 * 24 * 60 * 60 * 1000)}
      />
    )
  },
}

export const WithEvents: Story = {
  render: () => {
    const [date, setDate] = useState<Date | undefined>(new Date())
    const eventDates = [new Date(), new Date(Date.now() + 3 * 24 * 60 * 60 * 1000)]
    return (
      <Calendar
        mode="single"
        selected={date}
        onSelect={setDate}
        modifiers={{ event: eventDates }}
        modifiersClassNames={{ event: 'bg-primary/20' }}
      />
    )
  },
}
