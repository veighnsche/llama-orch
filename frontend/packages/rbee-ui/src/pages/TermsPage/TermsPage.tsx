import { TemplateContainer } from '@rbee/ui/molecules'
import { CTATemplate, FAQTemplate, HeroTemplate } from '@rbee/ui/templates'
import {
  termsCtaContainerProps,
  termsCtaProps,
  termsFaqContainerProps,
  termsFaqProps,
  termsHeroContainerProps,
  termsHeroProps,
} from './TermsPageProps'

/**
 * TermsPage - Legal page with Terms of Service
 *
 * Structure:
 * 1. HeroTemplate - Title, last updated, legal icon
 * 2. FAQTemplate - Terms sections as searchable Q&A
 * 3. CTATemplate - Contact legal team
 *
 * @returns Terms of Service page component
 */
export default function TermsPage() {
  return (
    <main>
      {/* Hero Section */}
      <TemplateContainer {...termsHeroContainerProps}>
        <HeroTemplate {...termsHeroProps} />
      </TemplateContainer>

      {/* Terms Content (FAQ Format) */}
      <TemplateContainer {...termsFaqContainerProps}>
        <FAQTemplate {...termsFaqProps} />
      </TemplateContainer>

      {/* Contact CTA */}
      <TemplateContainer {...termsCtaContainerProps}>
        <CTATemplate {...termsCtaProps} />
      </TemplateContainer>
    </main>
  )
}
