import { cn } from '@rbee/ui/utils'
import { focusRing } from '@rbee/ui/utils/focus-ring'
import type * as React from 'react'

function Textarea({ className, ...props }: React.ComponentProps<'textarea'>) {
  return (
    <textarea
      data-slot="textarea"
      className={cn(
        'border-input placeholder:text-slate-400 dark:placeholder:text-[#8b9bb0] bg-white dark:bg-[color:var(--background)] flex field-sizing-content min-h-16 w-full rounded-md border px-3 py-2 text-base transition-[color,box-shadow,border-color] disabled:cursor-not-allowed md:text-sm',
        '[box-shadow:inset_0_1px_0_rgba(15,23,42,0.04),0_1px_2px_rgba(15,23,42,0.04)]',
        'dark:[box-shadow:inset_0_1px_0_rgba(255,255,255,0.04),0_1px_2px_rgba(0,0,0,0.25)]',
        'hover:border-slate-400',
        'focus-visible:border-ring',
        'disabled:bg-[#1a2435] disabled:text-[#6c7a90] disabled:border-[#223047] disabled:placeholder:text-[#6c7a90]',
        focusRing,
        'aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive',
        className,
      )}
      {...props}
    />
  )
}

export { Textarea }
