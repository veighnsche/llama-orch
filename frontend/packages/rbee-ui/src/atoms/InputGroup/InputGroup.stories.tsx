import type { Meta, StoryObj } from '@storybook/react'
import { SearchIcon, XIcon } from 'lucide-react'
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput, InputGroupText } from './InputGroup'

const meta: Meta<typeof InputGroup> = {
  title: 'Atoms/InputGroup',
  component: InputGroup,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof InputGroup>

export const Default: Story = {
  render: () => (
    <InputGroup className="w-96">
      <InputGroupAddon>
        <SearchIcon />
      </InputGroupAddon>
      <InputGroupInput placeholder="Search..." />
    </InputGroup>
  ),
}

export const WithAddons: Story = {
  render: () => (
    <div className="flex flex-col gap-4 w-96">
      <InputGroup>
        <InputGroupAddon>
          <InputGroupText>@</InputGroupText>
        </InputGroupAddon>
        <InputGroupInput placeholder="Username" />
      </InputGroup>

      <InputGroup>
        <InputGroupInput placeholder="Amount" />
        <InputGroupAddon align="inline-end">
          <InputGroupText>.00</InputGroupText>
        </InputGroupAddon>
      </InputGroup>

      <InputGroup>
        <InputGroupAddon>
          <InputGroupText>https://</InputGroupText>
        </InputGroupAddon>
        <InputGroupInput placeholder="example.com" />
        <InputGroupAddon align="inline-end">
          <InputGroupText>.com</InputGroupText>
        </InputGroupAddon>
      </InputGroup>
    </div>
  ),
}

export const WithButtons: Story = {
  render: () => (
    <div className="flex flex-col gap-4 w-96">
      <InputGroup>
        <InputGroupAddon>
          <SearchIcon />
        </InputGroupAddon>
        <InputGroupInput placeholder="Search..." />
        <InputGroupAddon align="inline-end">
          <InputGroupButton>
            <XIcon />
          </InputGroupButton>
        </InputGroupAddon>
      </InputGroup>

      <InputGroup>
        <InputGroupInput placeholder="Enter email..." />
        <InputGroupAddon align="inline-end">
          <InputGroupButton variant="default" size="sm">
            Subscribe
          </InputGroupButton>
        </InputGroupAddon>
      </InputGroup>

      <InputGroup>
        <InputGroupAddon>
          <InputGroupButton size="icon-sm">
            <SearchIcon />
          </InputGroupButton>
        </InputGroupAddon>
        <InputGroupInput placeholder="Search..." />
        <InputGroupAddon align="inline-end">
          <InputGroupButton size="icon-sm">
            <XIcon />
          </InputGroupButton>
        </InputGroupAddon>
      </InputGroup>
    </div>
  ),
}

export const AllSizes: Story = {
  render: () => (
    <div className="flex flex-col gap-4 w-96">
      <InputGroup className="h-8">
        <InputGroupAddon>
          <SearchIcon className="size-3.5" />
        </InputGroupAddon>
        <InputGroupInput placeholder="Small input" className="text-xs" />
      </InputGroup>

      <InputGroup>
        <InputGroupAddon>
          <SearchIcon />
        </InputGroupAddon>
        <InputGroupInput placeholder="Default input" />
      </InputGroup>

      <InputGroup className="h-11">
        <InputGroupAddon>
          <SearchIcon className="size-5" />
        </InputGroupAddon>
        <InputGroupInput placeholder="Large input" className="text-base" />
      </InputGroup>
    </div>
  ),
}
