'use client'

import { GlassCard } from '@rbee/ui/atoms'
import { cn } from '@rbee/ui/utils'
import { useEffect, useState } from 'react'
import { KeyValuePair } from '../../atoms/KeyValuePair'

export interface FloatingKPICardProps {
  /** GPU Pool label and value */
  gpuPool?: { label: string; value: string }
  /** Cost label and value */
  cost?: { label: string; value: string }
  /** Latency label and value */
  latency?: { label: string; value: string }
  /** Additional CSS classes */
  className?: string
}

export function FloatingKPICard({
  gpuPool = { label: 'GPU Pool', value: '5 nodes / 8 GPUs' },
  cost = { label: 'Cost', value: '$0.00 / hr' },
  latency = { label: 'Latency', value: '~34 ms' },
  className,
}: FloatingKPICardProps) {
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
    <GlassCard
      className={cn(
        'absolute -bottom-16 left-[50%] p-4 space-y-2',
        'transition-all duration-300 z-10',
        isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2',
        className,
      )}
    >
      <KeyValuePair label={gpuPool.label} value={gpuPool.value} valueVariant="semibold" />
      <KeyValuePair label={cost.label} value={cost.value} valueVariant="success" />
      <KeyValuePair label={latency.label} value={latency.value} valueVariant="semibold" />
    </GlassCard>
  )
}
