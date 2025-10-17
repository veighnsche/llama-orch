import { CompliancePage } from '@rbee/ui/pages'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'rbee for Compliance - EU-Native, GDPR-Ready AI',
  description:
    'GDPR-compliant AI infrastructure by design. 7-year audit retention, EU data residency, zero US cloud dependencies. SOC2 and ISO 27001 aligned.',
  keywords: [
    'GDPR AI infrastructure',
    'EU data residency AI',
    'compliant AI platform',
    'SOC2 AI',
    'ISO 27001 AI',
    'audit retention',
  ],
  alternates: {
    canonical: '/industries/compliance',
  },
  openGraph: {
    title: 'rbee for Compliance - GDPR-Native AI Infrastructure',
    description: 'EU data residency, 7-year audit retention, SOC2 ready, ISO 27001 aligned.',
    type: 'website',
    url: 'https://rbee.dev/industries/compliance',
    siteName: 'rbee',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'rbee for Compliance - GDPR & SOC2',
    description: 'EU-native AI infrastructure. GDPR-compliant by design.',
  },
}

export default function Page() {
  return <CompliancePage />
}
