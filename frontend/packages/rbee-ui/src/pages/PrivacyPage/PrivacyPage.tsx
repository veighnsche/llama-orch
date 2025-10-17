'use client'

import { TemplateContainer } from '@rbee/ui/molecules'
import { CTATemplate, FAQTemplate, HeroTemplate } from '@rbee/ui/templates'
import { privacyCTAProps, privacyFAQProps, privacyHeroProps } from './PrivacyPageProps'

export default function PrivacyPage() {
  return (
    <main>
      <HeroTemplate {...privacyHeroProps} />
      <TemplateContainer
        title={null}
        background={{ variant: 'background' }}
        paddingY="2xl"
        maxWidth="7xl"
        align="center"
      >
        <FAQTemplate {...privacyFAQProps} />
      </TemplateContainer>
      <TemplateContainer
        title={null}
        background={{ variant: 'subtle-border' }}
        paddingY="xl"
        maxWidth="5xl"
        align="center"
      >
        <CTATemplate {...privacyCTAProps} />
      </TemplateContainer>
    </main>
  )
}
