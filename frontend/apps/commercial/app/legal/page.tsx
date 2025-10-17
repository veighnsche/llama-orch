import { LegalPage } from '@rbee/ui/pages'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'rbee Legal - Privacy Policy & Terms of Service',
  description: 'Legal information for rbee. Privacy policy, terms of service, and compliance documentation.',
  keywords: ['privacy policy', 'terms of service', 'legal', 'GDPR', 'compliance'],
  alternates: {
    canonical: '/legal',
  },
}

export default function Page() {
  return <LegalPage />
}
