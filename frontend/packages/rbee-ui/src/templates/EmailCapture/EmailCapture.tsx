'use client'

import { Badge } from '@rbee/ui/atoms/Badge'
import { Button } from '@rbee/ui/atoms/Button'
import { Input } from '@rbee/ui/atoms/Input'
import { BeeGlyph, HomelabBee } from '@rbee/ui/icons'
import { CheckCircle2, GitBranch, Lock, Mail } from 'lucide-react'
import type React from 'react'
import { useState } from 'react'

export interface EmailCaptureProps {
  /** Status badge configuration */
  badge?: {
    text: string
    showPulse?: boolean
  }
  /** Main headline */
  headline: string
  /** Subheadline / description */
  subheadline: string
  /** Email input configuration */
  emailInput: {
    placeholder: string
    label: string
  }
  /** Submit button configuration */
  submitButton: {
    label: string
  }
  /** Trust microcopy below the form */
  trustMessage: string
  /** Success state configuration */
  successMessage: string
  /** Community footer configuration */
  communityFooter?: {
    text: string
    linkText: string
    linkHref: string
    subtext: string
  }
  /** Show decorative bee glyphs */
  showBeeGlyphs?: boolean
  /** Show homelab illustration */
  showIllustration?: boolean
  /** Auto-reset delay in milliseconds (default: 3000) */
  autoResetDelay?: number
  /** Callback when email is submitted */
  onSubmit?: (email: string) => void | Promise<void>
}

export function EmailCapture({
  badge,
  headline,
  subheadline,
  emailInput,
  submitButton,
  trustMessage,
  successMessage,
  communityFooter,
  showBeeGlyphs = true,
  showIllustration = true,
  autoResetDelay = 3000,
  onSubmit,
}: EmailCaptureProps) {
  const [email, setEmail] = useState('')
  const [submitted, setSubmitted] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (onSubmit) {
      await onSubmit(email)
    } else {
      // Default behavior: log to console
      console.log('Email submitted:', email)
    }

    setSubmitted(true)
    setTimeout(() => {
      setSubmitted(false)
      setEmail('')
    }, autoResetDelay)
  }

  return (
    <section className="relative isolate py-28 bg-background">
      {/* Decorative bee glyphs */}
      {showBeeGlyphs && (
        <>
          <BeeGlyph className="absolute top-16 left-[8%] opacity-5 pointer-events-none" />
          <BeeGlyph className="absolute bottom-20 right-[10%] opacity-5 pointer-events-none" />
        </>
      )}

      <div className="relative max-w-3xl mx-auto px-6 text-center">
        {/* Status badge */}
        {badge && (
          <Badge
            variant="outline"
            className="mb-4 inline-flex items-center gap-2 border-primary/20 bg-primary/10 px-3 py-1 animate-in fade-in slide-in-from-bottom-2 duration-500"
            style={{ animationDelay: '100ms' }}
          >
            {badge.showPulse && (
              <span className="relative flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-75"></span>
                <span className="relative inline-flex h-2 w-2 rounded-full bg-primary"></span>
              </span>
            )}
            <span className="text-xs font-medium uppercase tracking-wider text-primary">{badge.text}</span>
          </Badge>
        )}

        {/* Headline */}
        <h2
          className="text-5xl md:text-6xl font-bold tracking-tight text-foreground mb-5 animate-in fade-in slide-in-from-bottom-2 duration-500"
          style={{ animationDelay: '300ms' }}
        >
          {headline}
        </h2>

        {/* Subhead */}
        <p
          className="text-lg md:text-xl text-muted-foreground mb-8 leading-relaxed max-w-2xl mx-auto animate-in fade-in slide-in-from-bottom-2 duration-500"
          style={{ animationDelay: '450ms' }}
        >
          {subheadline}
        </p>

        {/* Supportive visual - homelab illustration */}
        {showIllustration && (
          <HomelabBee
            size={960}
            className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-[52%] opacity-20"
            aria-hidden="true"
          />
        )}

        {/* Form or success state */}
        {!submitted ? (
          <form onSubmit={handleSubmit} className="mx-auto max-w-xl">
            <div className="flex flex-col sm:flex-row items-stretch gap-3">
              <div className="relative flex-1">
                <label htmlFor="waitlist-email" className="sr-only">
                  {emailInput.label}
                </label>
                <Mail
                  className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground/80"
                  aria-hidden="true"
                  focusable="false"
                />
                <Input
                  id="waitlist-email"
                  type="email"
                  placeholder={emailInput.placeholder}
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  className="h-12 pl-10 bg-card/80 border-border/70 text-foreground placeholder:text-muted-foreground focus-visible:ring-2 focus-visible:ring-primary/40 transition-shadow data-[invalid=true]:border-destructive/60 data-[invalid=true]:bg-destructive/5"
                />
              </div>
              <Button
                type="submit"
                className="h-12 px-7 bg-primary text-primary-foreground font-semibold rounded-xl shadow-sm hover:translate-y-[-1px] hover:shadow-md transition-transform"
              >
                {submitButton.label}
              </Button>
            </div>

            {/* Trust microcopy */}
            <div className="mt-3 text-sm text-muted-foreground flex items-center justify-center gap-2">
              <Lock className="w-3.5 h-3.5 text-muted-foreground/70" aria-hidden="true" focusable="false" />
              <span>{trustMessage}</span>
            </div>
          </form>
        ) : (
          <div
            className="inline-flex items-center gap-2 text-chart-3 text-base md:text-lg font-medium bg-card/60 border/60 rounded-xl px-4 py-3 shadow-xs"
            role="status"
            aria-live="polite"
          >
            <CheckCircle2 className="w-5 h-5" aria-hidden="true" focusable="false" />
            <span>{successMessage}</span>
          </div>
        )}

        {/* Community footer band */}
        {communityFooter && (
          <div className="mt-14 pt-8">
            {/* Gradient divider */}
            <div className="h-px w-full mx-auto bg-gradient-to-r from-transparent via-border to-transparent" />

            <p className="text-sm text-muted-foreground mt-6">{communityFooter.text}</p>

            <a
              href={communityFooter.linkHref}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 mt-3 text-primary font-medium hover:text-primary/90 transition-colors"
            >
              <GitBranch className="w-5 h-5" aria-hidden="true" focusable="false" />
              <span>{communityFooter.linkText}</span>
            </a>

            <p className="text-xs text-muted-foreground/80 mt-2">{communityFooter.subtext}</p>
          </div>
        )}
      </div>
    </section>
  )
}
