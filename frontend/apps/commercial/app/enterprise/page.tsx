import { EnterprisePage } from '@rbee/ui/pages'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'rbee Enterprise - GDPR-Compliant AI Infrastructure | SOC2 Ready',
  description:
    'Enterprise AI infrastructure with EU data residency, 7-year audit retention, zero US cloud dependencies. GDPR, SOC2, ISO 27001 compliant. Deploy on-premises or EU cloud.',
  keywords: [
    'GDPR compliant AI',
    'SOC2 AI infrastructure',
    'ISO 27001 AI',
    'EU data residency',
    'enterprise AI',
    'on-premises AI',
    'compliance AI',
    'audit trails',
    'private AI infrastructure',
  ],
  alternates: {
    canonical: '/enterprise',
  },
  openGraph: {
    title: 'rbee Enterprise - GDPR-Compliant AI Infrastructure',
    description:
      'EU data residency, 7-year audit retention, zero US cloud dependencies. SOC2 ready, ISO 27001 compliant.',
    type: 'website',
    url: 'https://rbee.dev/enterprise',
    siteName: 'rbee',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'rbee Enterprise - GDPR & SOC2 Compliant',
    description: 'Enterprise AI with EU data residency, complete compliance, zero US cloud dependencies.',
  },
}

export default function Enterprise() {
  return <EnterprisePage />
}
