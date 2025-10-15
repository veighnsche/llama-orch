// Created by: TEAM-007
import type { Meta, StoryObj } from '@storybook/react'
import { Button } from '../Button/Button'
import { Sheet, SheetContent, SheetDescription, SheetFooter, SheetHeader, SheetTitle, SheetTrigger } from './Sheet'

const meta: Meta<typeof Sheet> = {
  title: 'Atoms/Sheet',
  component: Sheet,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof Sheet>

/**
 * ## Overview
 * Sheet is a slide-out panel that overlays content from the edge of the screen.
 * Built on Radix UI Dialog primitive with directional animations.
 *
 * ## Composition
 * - Sheet (container)
 * - SheetTrigger (button to open)
 * - SheetContent (panel content)
 * - SheetHeader (title area)
 * - SheetTitle (heading)
 * - SheetDescription (subtitle)
 * - SheetFooter (action buttons)
 *
 * ## When to Use
 * - Mobile navigation menus
 * - Settings panels
 * - Filter controls
 * - Detail views
 *
 * ## Used In
 * - MobileNavigation (primary use case)
 */

export const Default: Story = {
  render: () => (
    <Sheet>
      <SheetTrigger asChild>
        <Button>Open Sheet</Button>
      </SheetTrigger>
      <SheetContent>
        <SheetHeader>
          <SheetTitle>Sheet Title</SheetTitle>
          <SheetDescription>Sheet description goes here</SheetDescription>
        </SheetHeader>
        <div className="py-4">
          <p className="text-sm">This is the main content area of the sheet.</p>
        </div>
      </SheetContent>
    </Sheet>
  ),
}

export const AllSides: Story = {
  render: () => (
    <div className="flex gap-4">
      <Sheet>
        <SheetTrigger asChild>
          <Button variant="outline">Left</Button>
        </SheetTrigger>
        <SheetContent side="left">
          <SheetHeader>
            <SheetTitle>Left Sheet</SheetTitle>
          </SheetHeader>
        </SheetContent>
      </Sheet>
      <Sheet>
        <SheetTrigger asChild>
          <Button variant="outline">Right</Button>
        </SheetTrigger>
        <SheetContent side="right">
          <SheetHeader>
            <SheetTitle>Right Sheet</SheetTitle>
          </SheetHeader>
        </SheetContent>
      </Sheet>
      <Sheet>
        <SheetTrigger asChild>
          <Button variant="outline">Top</Button>
        </SheetTrigger>
        <SheetContent side="top">
          <SheetHeader>
            <SheetTitle>Top Sheet</SheetTitle>
          </SheetHeader>
        </SheetContent>
      </Sheet>
      <Sheet>
        <SheetTrigger asChild>
          <Button variant="outline">Bottom</Button>
        </SheetTrigger>
        <SheetContent side="bottom">
          <SheetHeader>
            <SheetTitle>Bottom Sheet</SheetTitle>
          </SheetHeader>
        </SheetContent>
      </Sheet>
    </div>
  ),
}

export const WithContent: Story = {
  render: () => (
    <Sheet>
      <SheetTrigger asChild>
        <Button>Settings</Button>
      </SheetTrigger>
      <SheetContent>
        <SheetHeader>
          <SheetTitle>Settings</SheetTitle>
          <SheetDescription>Manage your account preferences</SheetDescription>
        </SheetHeader>
        <div className="py-6 space-y-4">
          <div>
            <label className="text-sm font-medium">Email Notifications</label>
            <p className="text-xs text-muted-foreground mt-1">Receive email updates</p>
          </div>
          <div>
            <label className="text-sm font-medium">Dark Mode</label>
            <p className="text-xs text-muted-foreground mt-1">Enable dark theme</p>
          </div>
        </div>
        <SheetFooter>
          <Button variant="outline">Cancel</Button>
          <Button>Save Changes</Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  ),
}

export const InMobileNav: Story = {
  render: () => (
    <Sheet>
      <SheetTrigger asChild>
        <Button variant="ghost" size="icon">
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </Button>
      </SheetTrigger>
      <SheetContent side="left">
        <SheetHeader>
          <SheetTitle>Menu</SheetTitle>
        </SheetHeader>
        <nav className="py-6">
          <ul className="space-y-4">
            <li>
              <a href="#" className="text-lg font-medium">
                Home
              </a>
            </li>
            <li>
              <a href="#" className="text-lg font-medium">
                Features
              </a>
            </li>
            <li>
              <a href="#" className="text-lg font-medium">
                Pricing
              </a>
            </li>
            <li>
              <a href="#" className="text-lg font-medium">
                Docs
              </a>
            </li>
          </ul>
        </nav>
        <SheetFooter>
          <Button className="w-full">Get Started</Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  ),
}
