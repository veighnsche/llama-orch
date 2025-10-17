'use client'

import * as CheckboxPrimitive from '@radix-ui/react-checkbox'
import { cn } from '@rbee/ui/utils'
import { focusRing } from '@rbee/ui/utils/focus-ring'
import { CheckIcon } from 'lucide-react'
import type * as React from 'react'

function Checkbox({ className, ...props }: React.ComponentProps<typeof CheckboxPrimitive.Root>) {
  return (
    <CheckboxPrimitive.Root
      data-slot="checkbox"
      className={cn(
        'peer border-input bg-white dark:bg-input/30 data-[state=checked]:bg-primary data-[state=checked]:text-primary-foreground dark:data-[state=checked]:bg-primary data-[state=checked]:border-primary hover:border-slate-400 size-4 shrink-0 rounded-[4px] border shadow-xs transition-all disabled:cursor-not-allowed disabled:opacity-50',
        'focus-visible:border-ring',
        focusRing,
        'aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive',
        className,
      )}
      {...props}
    >
      <CheckboxPrimitive.Indicator
        data-slot="checkbox-indicator"
        className="flex items-center justify-center text-current transition-none"
      >
        <CheckIcon className="size-3.5" />
      </CheckboxPrimitive.Indicator>
    </CheckboxPrimitive.Root>
  )
}

export { Checkbox }
