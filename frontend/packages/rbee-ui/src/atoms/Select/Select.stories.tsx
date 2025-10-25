// Created by: TEAM-008
import type { Meta, StoryObj } from '@storybook/react'
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from './Select'

const meta: Meta<typeof Select> = {
  title: 'Atoms/Select',
  component: Select,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
A select dropdown component built on Radix UI Select primitive.

## Features
- Single selection from dropdown list
- Keyboard navigation support
- Searchable options
- Grouped options with labels
- Two sizes: sm and default
- Accessible with proper ARIA attributes
- Portal-based dropdown positioning

## Used In
- Forms
- Filters
- Settings panels
- Data tables
				`,
      },
    },
  },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof Select>

export const Default: Story = {
  render: () => (
    <Select defaultValue="apple">
      <SelectTrigger className="w-[180px]">
        <SelectValue placeholder="Select a fruit" />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="apple">Apple</SelectItem>
        <SelectItem value="banana">Banana</SelectItem>
        <SelectItem value="orange">Orange</SelectItem>
        <SelectItem value="grape">Grape</SelectItem>
      </SelectContent>
    </Select>
  ),
}

export const WithGroups: Story = {
  render: () => (
    <Select defaultValue="llama-8b">
      <SelectTrigger className="w-[240px]">
        <SelectValue placeholder="Select a model" />
      </SelectTrigger>
      <SelectContent>
        <SelectGroup>
          <SelectLabel>Llama Models</SelectLabel>
          <SelectItem value="llama-8b">Llama 3.1 8B</SelectItem>
          <SelectItem value="llama-70b">Llama 3.1 70B</SelectItem>
          <SelectItem value="llama-405b">Llama 3.1 405B</SelectItem>
        </SelectGroup>
        <SelectGroup>
          <SelectLabel>Mistral Models</SelectLabel>
          <SelectItem value="mistral-7b">Mistral 7B</SelectItem>
          <SelectItem value="mistral-8x7b">Mixtral 8x7B</SelectItem>
        </SelectGroup>
        <SelectGroup>
          <SelectLabel>Other Models</SelectLabel>
          <SelectItem value="phi-3">Phi-3 Mini</SelectItem>
          <SelectItem value="gemma-7b">Gemma 7B</SelectItem>
        </SelectGroup>
      </SelectContent>
    </Select>
  ),
}

export const WithSearch: Story = {
  render: () => (
    <div className="flex flex-col gap-4">
      <div>
        <label className="text-sm font-medium mb-2 block">Deployment Region</label>
        <Select defaultValue="nl-ams">
          <SelectTrigger className="w-[280px]">
            <SelectValue placeholder="Select region" />
          </SelectTrigger>
          <SelectContent>
            <SelectGroup>
              <SelectLabel>Europe</SelectLabel>
              <SelectItem value="nl-ams">🇳🇱 Netherlands - Amsterdam</SelectItem>
              <SelectItem value="de-fra">🇩🇪 Germany - Frankfurt</SelectItem>
              <SelectItem value="fr-par">🇫🇷 France - Paris</SelectItem>
              <SelectItem value="uk-lon">🇬🇧 United Kingdom - London</SelectItem>
            </SelectGroup>
            <SelectGroup>
              <SelectLabel>North America</SelectLabel>
              <SelectItem value="us-east">🇺🇸 US East - Virginia</SelectItem>
              <SelectItem value="us-west">🇺🇸 US West - Oregon</SelectItem>
              <SelectItem value="ca-tor">🇨🇦 Canada - Toronto</SelectItem>
            </SelectGroup>
            <SelectGroup>
              <SelectLabel>Asia Pacific</SelectLabel>
              <SelectItem value="sg-sin">🇸🇬 Singapore</SelectItem>
              <SelectItem value="jp-tok">🇯🇵 Japan - Tokyo</SelectItem>
              <SelectItem value="au-syd">🇦🇺 Australia - Sydney</SelectItem>
            </SelectGroup>
          </SelectContent>
        </Select>
      </div>
    </div>
  ),
}

export const DisabledState: Story = {
  render: () => (
    <div className="flex flex-col gap-4">
      <div>
        <label className="text-sm font-medium mb-2 block">Enabled Select</label>
        <Select defaultValue="option-1">
          <SelectTrigger className="w-[200px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="option-1">Option 1</SelectItem>
            <SelectItem value="option-2">Option 2</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div>
        <label className="text-sm font-medium mb-2 block text-muted-foreground">Disabled Select</label>
        <Select defaultValue="option-1" disabled>
          <SelectTrigger className="w-[200px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="option-1">Option 1</SelectItem>
            <SelectItem value="option-2">Option 2</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </div>
  ),
}

export const Sizes: Story = {
  render: () => (
    <div className="flex flex-col gap-4">
      <div>
        <label className="text-sm font-medium mb-2 block">Small Size</label>
        <Select defaultValue="small">
          <SelectTrigger size="sm" className="w-[160px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="small">Small option</SelectItem>
            <SelectItem value="medium">Medium option</SelectItem>
            <SelectItem value="large">Large option</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div>
        <label className="text-sm font-medium mb-2 block">Default Size</label>
        <Select defaultValue="default">
          <SelectTrigger className="w-[200px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="default">Default option</SelectItem>
            <SelectItem value="another">Another option</SelectItem>
            <SelectItem value="third">Third option</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </div>
  ),
}

export const InForm: Story = {
  render: () => (
    <form className="w-full max-w-md space-y-6 p-6 border rounded">
      <div>
        <h3 className="text-lg font-semibold mb-2">Deploy Model</h3>
        <p className="text-sm text-muted-foreground">Configure your model deployment</p>
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">
          Model <span className="text-destructive">*</span>
        </label>
        <Select defaultValue="llama-8b">
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Select a model" />
          </SelectTrigger>
          <SelectContent>
            <SelectGroup>
              <SelectLabel>Recommended</SelectLabel>
              <SelectItem value="llama-8b">
                <div className="flex items-center gap-2">
                  <span>Llama 3.1 8B</span>
                  <span className="text-xs text-muted-foreground">Fast</span>
                </div>
              </SelectItem>
              <SelectItem value="mistral-7b">
                <div className="flex items-center gap-2">
                  <span>Mistral 7B</span>
                  <span className="text-xs text-muted-foreground">Balanced</span>
                </div>
              </SelectItem>
            </SelectGroup>
            <SelectGroup>
              <SelectLabel>Advanced</SelectLabel>
              <SelectItem value="llama-70b">Llama 3.1 70B</SelectItem>
              <SelectItem value="mixtral-8x7b">Mixtral 8x7B</SelectItem>
            </SelectGroup>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">Choose the model that fits your use case</p>
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">
          Region <span className="text-destructive">*</span>
        </label>
        <Select defaultValue="nl-ams">
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="nl-ams">🇳🇱 Netherlands - Amsterdam</SelectItem>
            <SelectItem value="de-fra">🇩🇪 Germany - Frankfurt</SelectItem>
            <SelectItem value="fr-par">🇫🇷 France - Paris</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">GPU Type</label>
        <Select defaultValue="a100">
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="a100">NVIDIA A100 (40GB)</SelectItem>
            <SelectItem value="a100-80">NVIDIA A100 (80GB)</SelectItem>
            <SelectItem value="h100">NVIDIA H100 (80GB)</SelectItem>
            <SelectItem value="l40s">NVIDIA L40S (48GB)</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">Replicas</label>
        <Select defaultValue="1">
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="1">1 replica</SelectItem>
            <SelectItem value="2">2 replicas</SelectItem>
            <SelectItem value="3">3 replicas</SelectItem>
            <SelectItem value="5">5 replicas</SelectItem>
            <SelectItem value="10">10 replicas</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">More replicas = higher availability</p>
      </div>

      <button
        type="submit"
        className="w-full px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90 transition-colors"
      >
        Deploy Model
      </button>
    </form>
  ),
}
