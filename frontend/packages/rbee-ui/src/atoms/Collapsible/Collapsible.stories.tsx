// Created by: TEAM-011

import { Button } from '@rbee/ui/atoms/Button'
import type { Meta, StoryObj } from '@storybook/react'
import { ChevronDown } from 'lucide-react'
import { useState } from 'react'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from './Collapsible'

const meta: Meta<typeof Collapsible> = {
  title: 'Atoms/Collapsible',
  component: Collapsible,
  parameters: { layout: 'centered' },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof Collapsible>

export const Default: Story = {
  render: () => {
    const [isOpen, setIsOpen] = useState(false)
    return (
      <Collapsible open={isOpen} onOpenChange={setIsOpen} className="w-[350px]">
        <div className="flex items-center justify-between border rounded-lg p-4">
          <h4 className="text-sm font-semibold">Collapsible Section</h4>
          <CollapsibleTrigger asChild>
            <Button variant="ghost" size="sm">
              <ChevronDown className={`h-4 w-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
            </Button>
          </CollapsibleTrigger>
        </div>
        <CollapsibleContent className="mt-2 border rounded-lg p-4">
          <p className="text-sm text-muted-foreground">This is the collapsible content.</p>
        </CollapsibleContent>
      </Collapsible>
    )
  },
}

export const Expanded: Story = {
  render: () => (
    <Collapsible defaultOpen className="w-[350px]">
      <div className="flex items-center justify-between border rounded-lg p-4">
        <h4 className="text-sm font-semibold">Expanded by Default</h4>
        <CollapsibleTrigger asChild>
          <Button variant="ghost" size="sm">
            <ChevronDown className="h-4 w-4" />
          </Button>
        </CollapsibleTrigger>
      </div>
      <CollapsibleContent className="mt-2 border rounded-lg p-4">
        <p className="text-sm text-muted-foreground">This content is visible by default.</p>
      </CollapsibleContent>
    </Collapsible>
  ),
}

export const WithTrigger: Story = {
  render: () => (
    <Collapsible className="w-[350px]">
      <CollapsibleTrigger asChild>
        <Button variant="outline" className="w-full justify-between">
          Click to expand
          <ChevronDown className="h-4 w-4" />
        </Button>
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-2 border rounded-lg p-4">
        <p className="text-sm">Content revealed when triggered.</p>
      </CollapsibleContent>
    </Collapsible>
  ),
}

export const Nested: Story = {
  render: () => (
    <Collapsible className="w-[350px]">
      <div className="border rounded-lg p-4">
        <CollapsibleTrigger asChild>
          <Button variant="ghost" className="w-full justify-between p-0">
            <span className="font-semibold">Parent Section</span>
            <ChevronDown className="h-4 w-4" />
          </Button>
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-4">
          <Collapsible>
            <CollapsibleTrigger asChild>
              <Button variant="ghost" size="sm" className="w-full justify-between">
                Child Section
                <ChevronDown className="h-4 w-4" />
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-2 pl-4">
              <p className="text-sm text-muted-foreground">Nested content</p>
            </CollapsibleContent>
          </Collapsible>
        </CollapsibleContent>
      </div>
    </Collapsible>
  ),
}
