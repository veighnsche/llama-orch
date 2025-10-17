'use client'

import { Badge, BrandMark, BrandWordmark, Button, Tooltip, TooltipContent, TooltipTrigger } from '@rbee/ui/atoms'
import { FeatureListItem, StatsGrid } from '@rbee/ui/molecules'
import { ArrowRight } from 'lucide-react'
import Image from 'next/image'
import type { ReactNode } from 'react'

export interface WhatIsRbeeFeature {
  icon: ReactNode
  title: string
  description: string
}

export interface WhatIsRbeeStat {
  value: string
  label: string
}

export interface WhatIsRbeeCTA {
  label: string
  href: string
  variant?: 'default' | 'ghost'
  showIcon?: boolean
}

export interface WhatIsRbeeProps {
  // Headline
  headlinePrefix: string
  headlineSuffix: string

  // Pronunciation
  pronunciationText: string
  pronunciationTooltip: string

  // Description
  description: string

  // Features
  features: WhatIsRbeeFeature[]

  // Stats
  stats: WhatIsRbeeStat[]

  // CTAs
  primaryCTA: WhatIsRbeeCTA
  secondaryCTA: WhatIsRbeeCTA

  // Closing copy
  closingCopyLine1: string
  closingCopyLine2: string

  // Visual
  visualImage: string
  visualImageAlt: string
  visualBadgeText: string
}

export function WhatIsRbee({
  headlinePrefix,
  headlineSuffix,
  pronunciationText,
  pronunciationTooltip,
  description,
  features,
  stats,
  primaryCTA,
  secondaryCTA,
  closingCopyLine1,
  closingCopyLine2,
  visualImage,
  visualImageAlt,
  visualBadgeText,
}: WhatIsRbeeProps) {
  return (
    <div className="max-w-5xl mx-auto">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-10 items-center">
        {/* Left column: Content */}
        <div className="space-y-6">
          {/* Headline with brand */}
          <h3 className="text-4xl md:text-5xl font-semibold text-foreground">
            <BrandMark size="xl" className="inline-block align-middle mr-3" />
            <BrandWordmark size="4xl" inline />
            {headlinePrefix}
          </h3>

          {/* Pronunciation badge */}
          <Tooltip>
            <TooltipTrigger asChild>
              <Badge
                variant="outline"
                className="text-xs cursor-help w-fit"
                aria-label={`rbee is ${pronunciationText}`}
              >
                {pronunciationText}
              </Badge>
            </TooltipTrigger>
            <TooltipContent>{pronunciationTooltip}</TooltipContent>
          </Tooltip>

          {/* Description */}
          <p className="text-lg text-muted-foreground">
            <BrandWordmark className="text-muted-foreground" /> {description}
          </p>

          {/* Value bullets */}
          <ul className="space-y-3">
            {features.map((feature, index) => (
              <FeatureListItem
                key={index}
                icon={feature.icon}
                title={feature.title}
                description={feature.description}
                iconColor="primary"
                iconVariant="rounded"
                iconSize="sm"
              />
            ))}
          </ul>

          {/* Stat cards grid */}
          <StatsGrid variant="cards" columns={3} className="pt-4" stats={stats} />

          {/* CTA row */}
          <div className="mt-6 flex flex-col sm:flex-row gap-3 justify-center md:justify-start">
            <Button size="lg" variant={primaryCTA.variant} asChild>
              <a href={primaryCTA.href}>
                {primaryCTA.label}
                {primaryCTA.showIcon && <ArrowRight className="ml-2 size-4" />}
              </a>
            </Button>
            <Button size="lg" variant={secondaryCTA.variant || 'ghost'} asChild>
              <a href={secondaryCTA.href} className="flex items-center gap-2">
                {secondaryCTA.label}
                {secondaryCTA.showIcon && <ArrowRight className="size-4" />}
              </a>
            </Button>
          </div>

          {/* Closing micro-copy */}
          <p className="text-base text-foreground leading-relaxed max-w-prose">
            {closingCopyLine1}
            <br />
            <span className="font-sans">{closingCopyLine2}</span>
          </p>
        </div>

        {/* Right column: Visual */}
        <div className="relative rounded-xl overflow-hidden ring-1 ring-border bg-card">
          {/* Network diagram image */}
          <div className="relative aspect-[16/9]">
            <div className="absolute top-4 left-4 z-10">
              <Badge variant="secondary" className="text-xs">
                {visualBadgeText}
              </Badge>
            </div>
            <Image
              src={visualImage}
              width={1280}
              height={720}
              priority
              className="w-full h-full object-cover will-change-transform motion-safe:transition-transform motion-safe:duration-500 motion-safe:hover:scale-[1.01]"
              alt={visualImageAlt}
            />
          </div>
        </div>
      </div>
    </div>
  )
}
