'use client'

import * as TabsPrimitive from '@radix-ui/react-tabs'
import { cn } from '@rbee/ui/utils'
import { focusRing } from '@rbee/ui/utils/focus-ring'
import type * as React from 'react'

function Tabs({ className, ...props }: React.ComponentProps<typeof TabsPrimitive.Root>) {
  return <TabsPrimitive.Root data-slot="tabs" className={cn('flex flex-col gap-2', className)} {...props} />
}

function TabsList({ className, ...props }: React.ComponentProps<typeof TabsPrimitive.List>) {
  return (
    <TabsPrimitive.List
      data-slot="tabs-list"
      className={cn(
        'inline-flex w-full lg:w-auto items-stretch lg:items-start justify-start rounded-none bg-transparent py-0 gap-2 overflow-x-auto lg:overflow-visible snap-x -mx-4 px-4 lg:mx-0 lg:px-0 lg:flex-col',
        className,
      )}
      {...props}
    />
  )
}

function TabsTrigger({ className, ...props }: React.ComponentProps<typeof TabsPrimitive.Trigger>) {
  return (
    <TabsPrimitive.Trigger
      data-slot="tabs-trigger"
      className={cn(
        'group inline-flex lg:flex w-max lg:w-full items-start justify-start gap-2 rounded-xl border border-transparent px-3 py-3 text-left text-sm font-medium snap-start text-foreground dark:text-muted-foreground transition-all hover:bg-muted/60 hover:-translate-y-0.5 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:bg-background data-[state=active]:border-border data-[state=active]:shadow-sm data-[state=active]:text-foreground [&_svg]:pointer-events-none [&_svg]:shrink-0',
        focusRing,
        className,
      )}
      {...props}
    />
  )
}

function TabsContent({ className, ...props }: React.ComponentProps<typeof TabsPrimitive.Content>) {
  return <TabsPrimitive.Content data-slot="tabs-content" className={cn('flex-1 outline-none', className)} {...props} />
}

export { Tabs, TabsList, TabsTrigger, TabsContent }
