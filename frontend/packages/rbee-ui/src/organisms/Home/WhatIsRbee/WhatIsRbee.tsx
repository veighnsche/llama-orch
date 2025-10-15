'use client'

import { homelabNetwork } from '@rbee/ui/assets'
import { Badge, BrandMark, BrandWordmark, Button, Tooltip, TooltipContent, TooltipTrigger } from '@rbee/ui/atoms'
import { FeatureListItem, SectionContainer, StatsGrid } from '@rbee/ui/molecules'
import { ArrowRight, Cpu, Shield, Zap } from 'lucide-react'
import Image from 'next/image'

export function WhatIsRbee() {
  return (
    <SectionContainer
      eyebrow={
        <Badge variant="secondary" className="uppercase tracking-wide">
          Open-source • Self-hosted
        </Badge>
      }
      title="What is rbee?"
      bgVariant="secondary"
      maxWidth="5xl"
      paddingY="xl"
      align="center"
    >
      <div className="max-w-5xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-10 items-center">
          {/* Left column: Content */}
          <div className="space-y-6">
            {/* Headline with brand */}
            <h3 className="text-4xl md:text-5xl font-semibold text-foreground">
              <BrandMark size="xl" className="inline-block align-middle mr-3" />
              <BrandWordmark size="4xl" inline />: your private AI infrastructure
            </h3>

            {/* Pronunciation badge */}
            <Tooltip>
              <TooltipTrigger asChild>
                <Badge variant="outline" className="text-xs cursor-help w-fit" aria-label="rbee is pronounced are-bee">
                  pronounced "are-bee"
                </Badge>
              </TooltipTrigger>
              <TooltipContent>Pronounced like "R.B."</TooltipContent>
            </Tooltip>

            {/* Please put rbee description here, muted font color */}
            <p className="text-lg text-muted-foreground">
              <BrandWordmark className="text-muted-foreground" /> is an open-source AI orchestration platform that
              unifies every computer in your home or office into a single, OpenAI-compatible AI cluster—private,
              controllable, and yours forever.
            </p>

            {/* Value bullets */}
            <ul className="space-y-3">
              <FeatureListItem
                icon={Zap}
                title="Independence"
                description="Build on your hardware. No surprise model or pricing changes."
                iconColor="primary"
                iconVariant="rounded"
                iconSize="sm"
              />
              <FeatureListItem
                icon={Shield}
                title="Privacy"
                description="Code and data never leave your network."
                iconColor="primary"
                iconVariant="rounded"
                iconSize="sm"
              />
              <FeatureListItem
                icon={Cpu}
                title="All GPUs together"
                description="CUDA, Metal, and CPU—scheduled as one."
                iconColor="primary"
                iconVariant="rounded"
                iconSize="sm"
              />
            </ul>

            {/* Stat cards grid */}
            <StatsGrid
              variant="cards"
              columns={3}
              className="pt-4"
              stats={[
                { value: '$0', label: 'No API fees' },
                { value: '100%', label: 'Private' },
                { value: 'All', label: 'CUDA · Metal · CPU' },
              ]}
            />

            {/* CTA row */}
            <div className="mt-6 flex flex-col sm:flex-row gap-3 justify-center md:justify-start">
              <Button size="lg" asChild>
                <a href="#get-started">Get Started Free</a>
              </Button>
              <Button variant="ghost" size="lg" asChild>
                <a href="/technical-deep-dive" className="flex items-center gap-2">
                  See Architecture
                  <ArrowRight className="size-4" />
                </a>
              </Button>
            </div>

            {/* Closing micro-copy */}
            <p className="text-base text-foreground leading-relaxed max-w-prose">
              OpenAI-compatible API. Zed/Cursor-ready.
              <br />
              <span className="font-sans">Your models, your rules.</span>
            </p>
          </div>

          {/* Right column: Visual */}
          <div className="relative rounded-xl overflow-hidden ring-1 ring-border bg-card">
            {/* Network diagram image */}
            <div className="relative aspect-[16/9]">
              <div className="absolute top-4 left-4 z-10">
                <Badge variant="secondary" className="text-xs">
                  Local Network
                </Badge>
              </div>
              <Image
                src={homelabNetwork}
                width={1280}
                height={720}
                priority
                className="w-full h-full object-cover will-change-transform motion-safe:transition-transform motion-safe:duration-500 motion-safe:hover:scale-[1.01]"
                alt="Distributed homelab AI network diagram showing a central orchestrator mini PC coordinating multiple worker nodes (gaming PCs, workstation, Mac Studio) connected via ethernet cables in a star topology. Each node displays GPU utilization and the network shows zero-cost local inference."
              />
            </div>
          </div>
        </div>
      </div>
    </SectionContainer>
  )
}
