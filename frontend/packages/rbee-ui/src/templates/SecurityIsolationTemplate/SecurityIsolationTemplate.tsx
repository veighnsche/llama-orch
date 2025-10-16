'use client'

import { Badge, Card, CardContent } from '@rbee/ui/atoms'
import { BulletListItem, CrateCard, IconCardHeader } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import { Lock, Shield } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import type { ReactNode } from 'react'

export interface SecurityCrate {
  name: string
  description: string
  hoverColor: string
}

export interface IsolationFeature {
  title: string
  color: 'chart-3' | 'chart-2'
}

export interface SecurityIsolationTemplateProps {
  cratesTitle: string
  cratesSubtitle: string
  securityCrates: SecurityCrate[]
  processIsolationTitle: string
  processIsolationSubtitle: string
  processFeatures: IsolationFeature[]
  zeroTrustTitle: string
  zeroTrustSubtitle: string
  zeroTrustFeatures: IsolationFeature[]
  className?: string
}

export function SecurityIsolationTemplate({
  cratesTitle,
  cratesSubtitle,
  securityCrates,
  processIsolationTitle,
  processIsolationSubtitle,
  processFeatures,
  zeroTrustTitle,
  zeroTrustSubtitle,
  zeroTrustFeatures,
  className,
}: SecurityIsolationTemplateProps) {
  return (
    <div className={cn('', className)}>
      <div className="mx-auto max-w-6xl space-y-8">
          <Card className="animate-in fade-in slide-in-from-bottom-2">
            <CardContent className="p-6 md:p-8">
              <IconCardHeader
                icon={Shield}
                iconTone="chart-2"
                iconSize="md"
                title={cratesTitle}
                subtitle={cratesSubtitle}
                useCardHeader={false}
                className="mb-6"
              />

              <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {securityCrates.map((crate) => (
                  <CrateCard
                    key={crate.name}
                    name={crate.name}
                    description={crate.description}
                    hoverColor={crate.hoverColor}
                  />
                ))}
              </div>
            </CardContent>
          </Card>

          <div className="grid md:grid-cols-2 gap-6 animate-in fade-in slide-in-from-bottom-2 delay-100">
            <Card>
              <CardContent className="p-6">
                <IconCardHeader
                  icon={Lock}
                  iconTone="chart-3"
                  iconSize="sm"
                  title={processIsolationTitle}
                  subtitle={processIsolationSubtitle}
                  titleClassName="text-lg"
                  subtitleClassName="text-sm mt-1"
                  useCardHeader={false}
                  className="mb-4"
                />
                <ul className="space-y-2">
                  {processFeatures.map((feature, idx) => (
                    <BulletListItem
                      key={idx}
                      title={feature.title}
                      color={feature.color}
                      variant="dot"
                      showPlate={false}
                    />
                  ))}
                </ul>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <IconCardHeader
                  icon={Shield}
                  iconTone="chart-2"
                  iconSize="sm"
                  title={zeroTrustTitle}
                  subtitle={zeroTrustSubtitle}
                  titleClassName="text-lg"
                  subtitleClassName="text-sm mt-1"
                  useCardHeader={false}
                  className="mb-4"
                />
                <ul className="space-y-2">
                  {zeroTrustFeatures.map((feature, idx) => (
                    <BulletListItem
                      key={idx}
                      title={feature.title}
                      color={feature.color}
                      variant="dot"
                      showPlate={false}
                    />
                  ))}
                </ul>
              </CardContent>
            </Card>
        </div>
      </div>
    </div>
  )
}
