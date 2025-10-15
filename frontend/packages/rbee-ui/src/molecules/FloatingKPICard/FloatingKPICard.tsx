'use client'

import { cn } from '@rbee/ui/utils'
import { useEffect, useState } from 'react'
import { KeyValuePair } from '../../atoms/KeyValuePair'

export interface FloatingKPICardProps {
  /** Additional CSS classes */
  className?: string
}

export function FloatingKPICard({ className }: FloatingKPICardProps) {
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    // Respect prefers-reduced-motion
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches

    if (prefersReducedMotion) {
      setIsVisible(true)
    } else {
      const timer = setTimeout(() => setIsVisible(true), 150)
      return () => clearTimeout(timer)
    }
  }, [])

  return (
    <div
      className={cn(
        'absolute -bottom-16 left-[50%] rounded-2xl shadow-lg/40 backdrop-blur-md',
        'bg-secondary/60 dark:bg-secondary/30 p-4 space-y-2',
        'transition-all duration-300 z-10',
        isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2',
        className,
      )}
    >
      <KeyValuePair label="GPU Pool" value="5 nodes / 8 GPUs" valueVariant="semibold" />
      <KeyValuePair label="Cost" value="$0.00 / hr" valueVariant="success" />
      <KeyValuePair label="Latency" value="~34 ms" valueVariant="semibold" />
    </div>
  )
}
