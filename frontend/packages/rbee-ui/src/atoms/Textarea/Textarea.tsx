import { cn } from '@rbee/ui/utils'
import { focusRing } from '@rbee/ui/utils/focus-ring'
import type * as React from 'react'

function Textarea({ className, ...props }: React.ComponentProps<'textarea'>) {
  return (
    <textarea
      data-slot="textarea"
      className={cn(
        'border-input placeholder:text-slate-400 bg-white dark:bg-input/30 flex field-sizing-content min-h-16 w-full rounded-md border px-3 py-2 text-base transition-[color,box-shadow,border-color] disabled:cursor-not-allowed disabled:opacity-50 md:text-sm',
        '[box-shadow:inset_0_1px_0_rgba(15,23,42,0.04),0_1px_2px_rgba(15,23,42,0.04)]',
        'hover:border-slate-400',
        'focus-visible:border-ring',
        focusRing,
        'aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive',
        className,
      )}
      {...props}
    />
  )
}

export { Textarea }
