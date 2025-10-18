import { LegalPage } from '@rbee/ui/pages'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'rbee for Legal - AI-Powered Legal Research for Law Firms',
  description:
    'Transform legal workflows with private AI infrastructure. Process contracts, case law, and discovery documents at scale—without sending client data to third-party APIs. Maintains attorney-client privilege.',
  keywords: [
    'legal AI',
    'law firm AI',
    'attorney-client privilege AI',
    'contract review AI',
    'legal research AI',
    'document review AI',
    'ABA compliant AI',
    'on-premises legal AI',
  ],
  alternates: {
    canonical: '/industries/legal',
  },
  openGraph: {
    title: 'rbee for Legal - Private AI for Law Firms',
    description:
      'AI-powered legal research and document review that preserves attorney-client privilege. 100% on-premises processing.',
    type: 'website',
    url: 'https://rbee.dev/industries/legal',
    siteName: 'rbee',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'rbee for Legal - Attorney-Client Privilege Protected',
    description: 'Private AI for contract review, legal research, and discovery analysis.',
  },
}

export default function Page() {
  return <LegalPage />
}
