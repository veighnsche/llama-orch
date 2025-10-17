// Dark Mode Polish Showcase for Overlays (Dialog/Popover)
import type { Meta, StoryObj } from '@storybook/react'
import { useState } from 'react'
import { Button } from '../Button/Button'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from './Dialog'
import { Popover, PopoverContent, PopoverTrigger } from '../Popover/Popover'

const meta: Meta<typeof Dialog> = {
  title: 'Atoms/Overlays/Dark Mode Polish',
  component: Dialog,
  parameters: {
    layout: 'padded',
    backgrounds: {
      default: 'dark',
      values: [{ name: 'dark', value: '#0b1220' }],
    },
  },
  decorators: [
    (Story) => (
      <div className="dark">
        <Story />
      </div>
    ),
  ],
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof Dialog>

/**
 * ## Dark Mode Polish: Overlays
 * 
 * Demonstrates:
 * - Scrim: bg-black/60 (reduced contrast jump)
 * - Blur: backdrop-blur-[2px] (improves readability behind overlay)
 * - Motion: fade-in-50/fade-out-50 (soft, symmetric)
 * - Shadows: var(--shadow-lg) for Dialog, var(--shadow-md) for Popover
 */

export const DialogWithScrim: Story = {
  render: () => {
    const [open, setOpen] = useState(false)
    
    return (
      <div className="bg-[#0b1220] p-8 rounded-lg min-h-[400px]">
        <div className="bg-card border border-border rounded-xl p-6 max-w-2xl">
          <h3 className="text-lg font-semibold mb-4">Dialog Overlay</h3>
          <p className="text-sm text-muted-foreground mb-4">
            Click the button to see the scrim + blur effect. The backdrop uses bg-black/60 backdrop-blur-[2px] to reduce
            perceived contrast jump and improve readability of content behind the overlay.
          </p>
          
          <Dialog open={open} onOpenChange={setOpen}>
            <DialogTrigger asChild>
              <Button>Open Dialog</Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Scrim + Blur Overlay</DialogTitle>
                <DialogDescription>
                  Notice the subtle blur on the background content. This reduces the harsh contrast jump when the dialog opens.
                </DialogDescription>
              </DialogHeader>
              <div className="py-4">
                <p className="text-sm">
                  The overlay uses:
                </p>
                <ul className="list-disc list-inside text-sm space-y-1 mt-2 text-muted-foreground">
                  <li>bg-black/60 (darker than /50 for better separation)</li>
                  <li>backdrop-blur-[2px] (subtle blur, not heavy)</li>
                  <li>fade-in-50/fade-out-50 (smooth motion)</li>
                  <li>var(--shadow-lg) for dialog panel</li>
                </ul>
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={() => setOpen(false)}>
                  Close
                </Button>
                <Button onClick={() => setOpen(false)}>Confirm</Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>

          <div className="mt-6 p-4 bg-muted rounded-md">
            <p className="text-sm font-medium mb-2">Background Content</p>
            <p className="text-xs text-muted-foreground">
              This content will be visible (but blurred) behind the dialog overlay. The blur effect helps maintain
              context while focusing attention on the modal.
            </p>
          </div>
        </div>
      </div>
    )
  },
  parameters: {
    docs: {
      description: {
        story: 'Dialog with scrim + blur overlay showing reduced contrast jump and improved readability.',
      },
    },
  },
}

export const PopoverWithShadow: Story = {
  render: () => (
    <div className="bg-[#0b1220] p-8 rounded-lg min-h-[400px]">
      <div className="bg-card border border-border rounded-xl p-6 max-w-2xl">
        <h3 className="text-lg font-semibold mb-4">Popover Overlay</h3>
        <p className="text-sm text-muted-foreground mb-4">
          Popovers use var(--shadow-md) which includes ambient shadow + subtle highlight inset for depth on dark.
        </p>
        
        <div className="flex gap-4">
          <Popover>
            <PopoverTrigger asChild>
              <Button variant="outline">Open Popover</Button>
            </PopoverTrigger>
            <PopoverContent>
              <div className="space-y-2">
                <h4 className="font-semibold">Popover Title</h4>
                <p className="text-sm text-muted-foreground">
                  This popover uses the refined dark shadow with ambient + highlight inset:
                </p>
                <code className="text-xs block mt-2 p-2 bg-muted rounded">
                  0 6px 16px rgba(0,0,0,0.45),<br />
                  0 1px 0 rgba(255,255,255,0.04) inset
                </code>
              </div>
            </PopoverContent>
          </Popover>

          <Popover>
            <PopoverTrigger asChild>
              <Button>Another Popover</Button>
            </PopoverTrigger>
            <PopoverContent>
              <div className="space-y-2">
                <h4 className="font-semibold">Motion Parity</h4>
                <p className="text-sm text-muted-foreground">
                  Open/close animations use fade-in-50/fade-out-50 + slide-in-from-top-1 for soft, symmetric motion.
                </p>
              </div>
            </PopoverContent>
          </Popover>
        </div>

        <div className="mt-6 p-4 bg-muted rounded-md">
          <p className="text-sm font-medium mb-2">Shadow Comparison</p>
          <div className="space-y-2 text-xs text-muted-foreground">
            <p><strong>Dialog:</strong> var(--shadow-lg) — Heavier for full-screen modals</p>
            <p><strong>Popover:</strong> var(--shadow-md) — Medium for floating panels</p>
            <p><strong>Card:</strong> var(--shadow-sm) — Light for surface elevation</p>
          </div>
        </div>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Popover showing refined dark shadows with ambient + highlight insets.',
      },
    },
  },
}

export const MotionComparison: Story = {
  render: () => {
    const [dialogOpen, setDialogOpen] = useState(false)
    
    return (
      <div className="bg-[#0b1220] p-8 rounded-lg min-h-[400px]">
        <div className="bg-card border border-border rounded-xl p-6 max-w-2xl">
          <h3 className="text-lg font-semibold mb-4">Motion Choreography</h3>
          <p className="text-sm text-muted-foreground mb-4">
            All overlays use consistent motion: fade-in-50/fade-out-50 + directional slide. This creates soft,
            symmetric transitions that respect prefers-reduced-motion.
          </p>
          
          <div className="space-y-4">
            <div>
              <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
                <DialogTrigger asChild>
                  <Button>Open Dialog (Watch Motion)</Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Motion Test</DialogTitle>
                    <DialogDescription>
                      Notice the smooth fade + zoom in/out. The overlay fades in at 50% opacity for a softer entrance.
                    </DialogDescription>
                  </DialogHeader>
                  <div className="py-4">
                    <p className="text-sm text-muted-foreground">
                      Close and reopen to observe the symmetric motion. Both entrance and exit feel balanced.
                    </p>
                  </div>
                  <DialogFooter>
                    <Button onClick={() => setDialogOpen(false)}>Close</Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            </div>

            <div className="p-4 bg-muted rounded-md">
              <p className="text-sm font-medium mb-2">Technical Details</p>
              <ul className="list-disc list-inside text-xs space-y-1 text-muted-foreground">
                <li>data-[state=open]:animate-in fade-in-50</li>
                <li>data-[state=closed]:animate-out fade-out-50</li>
                <li>zoom-in-95/zoom-out-95 for Dialog</li>
                <li>slide-in-from-top-1 for Popover/Dropdown</li>
                <li>Respects prefers-reduced-motion</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    )
  },
  parameters: {
    docs: {
      description: {
        story: 'Motion choreography showing soft, symmetric transitions for all overlay types.',
      },
    },
  },
}
