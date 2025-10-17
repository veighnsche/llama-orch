'use client'

import { Alert, AlertDescription, AlertTitle, Badge, Card, CardContent } from '@rbee/ui/atoms'
import { IconCardHeader, TerminalWindow } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import { AlertTriangle, CheckCircle2, Cpu, X } from 'lucide-react'
import type { ReactNode } from 'react'

export interface PolicyBadge {
  label: string
  variant: 'destructive' | 'success'
}

export interface BackendDetection {
  label: string
  variant: 'primary' | 'muted' | 'success'
}

export interface FeatureCard {
  icon: ReactNode
  title: string
  description: string
}

export interface MultiBackendGpuTemplateProps {
  /** Policy title */
  policyTitle: string
  /** Policy subtitle */
  policySubtitle: string
  /** Prohibited badges */
  prohibitedBadges: PolicyBadge[]
  /** What happens badges */
  whatHappensBadges: PolicyBadge[]
  /** Error alert title */
  errorTitle: string
  /** Error suggestions list */
  errorSuggestions: string[]
  /** Terminal title */
  terminalTitle: string
  /** Terminal content */
  terminalContent: ReactNode
  /** Backend detections */
  backendDetections: BackendDetection[]
  /** Total devices count */
  totalDevices: number
  /** Terminal footer note */
  terminalFooter: string
  /** Feature cards (3 items) */
  featureCards: FeatureCard[]
  /** Custom class name */
  className?: string
}

/**
 * MultiBackendGpuTemplate - Template for multi-backend GPU support features
 *
 * @example
 * ```tsx
 * <MultiBackendGpuTemplate
 *   title="Multi-Backend GPU Support"
 *   subtitle="CUDA, Metal, and CPU backends with explicit device selection."
 *   policyTitle="GPU FAIL FAST policy"
 *   featureCards={[...]}
 * />
 * ```
 */
export function MultiBackendGpuTemplate({
  policyTitle,
  policySubtitle,
  prohibitedBadges,
  whatHappensBadges,
  errorTitle,
  errorSuggestions,
  terminalTitle,
  terminalContent,
  backendDetections,
  totalDevices,
  terminalFooter,
  featureCards,
  className,
}: MultiBackendGpuTemplateProps) {
  return (
    <div className={cn('', className)}>
      <div className="mx-auto max-w-6xl space-y-10">
        {/* 1. Policy Poster (full-bleed branded banner) */}
        <Card className="relative overflow-hidden border-primary/40 bg-gradient-to-b from-primary/10 to-background animate-in fade-in slide-in-from-bottom-2">
          <CardContent className="p-8 md:p-10 space-y-6">
            <IconCardHeader
              icon={<AlertTriangle className="w-6 h-6" />}
              title={policyTitle}
              subtitle={policySubtitle}
              iconTone="primary"
              iconSize="md"
              titleClassName="text-3xl md:text-4xl font-extrabold"
              subtitleClassName="text-lg"
            />

            {/* Prohibited pills (red) */}
            <div>
              <div className="text-sm font-semibold text-muted-foreground mb-2">Prohibited:</div>
              <div className="flex flex-wrap gap-2">
                {prohibitedBadges.map((badge, idx) => (
                  <Badge key={idx} variant="destructive">
                    {badge.label}
                  </Badge>
                ))}
              </div>
            </div>

            {/* What happens pills (green) */}
            <div>
              <div className="text-sm font-semibold text-muted-foreground mb-2">What happens:</div>
              <div className="flex flex-wrap gap-2">
                {whatHappensBadges.map((badge, idx) => (
                  <SuccessBadge key={idx}>{badge.label}</SuccessBadge>
                ))}
              </div>
            </div>

            {/* Inline error alert */}
            <Alert variant="destructive">
              <X />
              <AlertTitle className="font-mono">{errorTitle}</AlertTitle>
              <AlertDescription>
                <ul className="list-disc pl-5 space-y-1">
                  {errorSuggestions.map((suggestion, idx) => (
                    <li key={idx}>{suggestion}</li>
                  ))}
                </ul>
              </AlertDescription>
            </Alert>
          </CardContent>
        </Card>

        {/* 2. Detection Console (wide terminal) */}
        <TerminalWindow
          showChrome
          title={terminalTitle}
          className="animate-in fade-in slide-in-from-bottom-2 delay-100"
        >
          {terminalContent}
          <div className="mt-3 text-muted-foreground">Available backends:</div>
          <div className="mt-2 flex flex-wrap gap-2">
            {backendDetections.map((detection, idx) => (
              <BackendBadge key={idx} variant={detection.variant}>
                {detection.label}
              </BackendBadge>
            ))}
          </div>
          <div className="mt-4 text-foreground">Total devices: {totalDevices}</div>
          <div className="mt-4 pt-4 border-t border-border text-sm text-muted-foreground">{terminalFooter}</div>
        </TerminalWindow>

        {/* 3. Microcards strip (3-up) */}
        <div className="grid sm:grid-cols-3 gap-3 animate-in fade-in slide-in-from-bottom-2 delay-150">
          {featureCards.map((card, idx) => (
            <div
              key={idx}
              className="bg-background rounded-xl border border-border p-4 flex items-start gap-3 hover:-translate-y-0.5 transition-transform"
            >
              <div className="size-5 shrink-0 mt-0.5 text-chart-2" aria-hidden="true">
                {card.icon}
              </div>
              <div>
                <div className="font-semibold text-foreground text-sm">{card.title}</div>
                <div className="text-xs text-muted-foreground mt-1">{card.description}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// Helper Components
interface SuccessBadgeProps {
  children: ReactNode
}

function SuccessBadge({ children }: SuccessBadgeProps) {
  return (
    <span className="inline-flex items-center gap-2 rounded-full bg-chart-3/10 text-chart-3 px-3 py-1 text-xs font-semibold">
      {children}
    </span>
  )
}

interface BackendBadgeProps {
  variant: 'primary' | 'muted' | 'success'
  children: ReactNode
}

function BackendBadge({ variant, children }: BackendBadgeProps) {
  const variantClasses = {
    primary: 'bg-primary/10 text-primary',
    muted: 'bg-muted text-foreground/80',
    success: 'bg-emerald-500/10 text-emerald-400',
  }

  return <span className={cn('rounded-md px-2 py-1 text-xs font-semibold', variantClasses[variant])}>{children}</span>
}
