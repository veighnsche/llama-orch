'use client'

import { TemplateContainer } from '@rbee/ui/molecules'
import {
  AdditionalFeaturesGrid,
  CTATemplate,
  EmailCapture,
  EnterpriseCompliance,
  EnterpriseHowItWorks,
  FAQTemplate,
  HeroTemplate,
  HowItWorks,
  TestimonialsTemplate,
  UseCasesTemplate,
} from '@rbee/ui/templates'
import {
  communityCTAContainerProps,
  communityCTAProps,
  communityEmailCaptureContainerProps,
  communityEmailCaptureProps,
  communityFAQContainerProps,
  communityFAQProps,
  communityGuidelinesContainerProps,
  communityGuidelinesProps,
  communityHeroContainerProps,
  communityHeroProps,
  CommunityStats,
  communityStatsContainerProps,
  contributionTypesContainerProps,
  contributionTypesProps,
  featuredContributorsContainerProps,
  featuredContributorsProps,
  howToContributeContainerProps,
  howToContributeProps,
  roadmapContainerProps,
  roadmapProps,
  supportChannelsContainerProps,
  supportChannelsProps,
} from './CommunityPageProps'

export default function CommunityPage() {
  return (
    <>
      {/* Hero Section */}
      <TemplateContainer {...communityHeroContainerProps}>
        <HeroTemplate {...communityHeroProps} />
      </TemplateContainer>

      {/* Email Capture */}
      <TemplateContainer {...communityEmailCaptureContainerProps}>
        <EmailCapture {...communityEmailCaptureProps} />
      </TemplateContainer>

      {/* Community Stats */}
      <TemplateContainer {...communityStatsContainerProps}>
        <CommunityStats />
      </TemplateContainer>

      {/* Contribution Types */}
      <TemplateContainer {...contributionTypesContainerProps}>
        <UseCasesTemplate {...contributionTypesProps} />
      </TemplateContainer>

      {/* How to Contribute */}
      <TemplateContainer {...howToContributeContainerProps}>
        <HowItWorks {...howToContributeProps} />
      </TemplateContainer>

      {/* Support Channels */}
      <TemplateContainer {...supportChannelsContainerProps}>
        <AdditionalFeaturesGrid {...supportChannelsProps} />
      </TemplateContainer>

      {/* Community Guidelines */}
      <TemplateContainer {...communityGuidelinesContainerProps}>
        <EnterpriseCompliance {...communityGuidelinesProps} />
      </TemplateContainer>

      {/* Featured Contributors */}
      <TemplateContainer {...featuredContributorsContainerProps}>
        <TestimonialsTemplate {...featuredContributorsProps} />
      </TemplateContainer>

      {/* Roadmap */}
      <TemplateContainer {...roadmapContainerProps}>
        <EnterpriseHowItWorks {...roadmapProps} />
      </TemplateContainer>

      {/* FAQ */}
      <TemplateContainer {...communityFAQContainerProps}>
        <FAQTemplate {...communityFAQProps} />
      </TemplateContainer>

      {/* Final CTA */}
      <TemplateContainer {...communityCTAContainerProps}>
        <CTATemplate {...communityCTAProps} />
      </TemplateContainer>
    </>
  )
}
