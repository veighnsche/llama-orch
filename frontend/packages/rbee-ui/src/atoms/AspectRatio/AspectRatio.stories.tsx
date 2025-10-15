// Created by: TEAM-011
import type { Meta, StoryObj } from '@storybook/react'
import { AspectRatio } from './AspectRatio'

const meta: Meta<typeof AspectRatio> = {
  title: 'Atoms/AspectRatio',
  component: AspectRatio,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  argTypes: {
    ratio: {
      control: 'number',
      description: 'Aspect ratio (width / height)',
    },
  },
}

export default meta
type Story = StoryObj<typeof AspectRatio>

/**
 * ## Overview
 * AspectRatio maintains a consistent width-to-height ratio for content,
 * useful for responsive images and videos. Built with Radix UI primitives.
 *
 * ## When to Use
 * - Display responsive images
 * - Embed videos
 * - Create consistent card layouts
 * - Maintain aspect ratios across viewports
 *
 * ## Used In
 * - Image galleries
 * - Video players
 * - Card components
 * - Media previews
 */

export const Default: Story = {
  render: () => (
    <div className="w-[450px]">
      <AspectRatio ratio={16 / 9}>
        <div className="flex h-full w-full items-center justify-center rounded-md bg-muted">
          <span className="text-sm text-muted-foreground">16:9 Aspect Ratio</span>
        </div>
      </AspectRatio>
    </div>
  ),
}

export const AllRatios: Story = {
  render: () => (
    <div className="flex flex-col gap-6 w-[450px]">
      <div>
        <p className="text-sm font-medium mb-2">16:9 (Widescreen)</p>
        <AspectRatio ratio={16 / 9}>
          <div className="flex h-full w-full items-center justify-center rounded-md bg-muted">
            <span className="text-sm text-muted-foreground">16:9</span>
          </div>
        </AspectRatio>
      </div>
      <div>
        <p className="text-sm font-medium mb-2">4:3 (Standard)</p>
        <AspectRatio ratio={4 / 3}>
          <div className="flex h-full w-full items-center justify-center rounded-md bg-muted">
            <span className="text-sm text-muted-foreground">4:3</span>
          </div>
        </AspectRatio>
      </div>
      <div>
        <p className="text-sm font-medium mb-2">1:1 (Square)</p>
        <AspectRatio ratio={1}>
          <div className="flex h-full w-full items-center justify-center rounded-md bg-muted">
            <span className="text-sm text-muted-foreground">1:1</span>
          </div>
        </AspectRatio>
      </div>
      <div>
        <p className="text-sm font-medium mb-2">21:9 (Ultrawide)</p>
        <AspectRatio ratio={21 / 9}>
          <div className="flex h-full w-full items-center justify-center rounded-md bg-muted">
            <span className="text-sm text-muted-foreground">21:9</span>
          </div>
        </AspectRatio>
      </div>
    </div>
  ),
}

export const WithImage: Story = {
  render: () => (
    <div className="w-[450px]">
      <AspectRatio ratio={16 / 9}>
        <img
          src="https://images.unsplash.com/photo-1588345921523-c2dcdb7f1dcd?w=800&dpr=2&q=80"
          alt="Photo by Drew Beamer"
          className="h-full w-full rounded-md object-cover"
        />
      </AspectRatio>
    </div>
  ),
}

export const WithVideo: Story = {
  render: () => (
    <div className="w-[450px]">
      <AspectRatio ratio={16 / 9}>
        <iframe
          src="https://www.youtube.com/embed/dQw4w9WgXcQ"
          title="YouTube video player"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
          className="h-full w-full rounded-md"
        />
      </AspectRatio>
    </div>
  ),
}
