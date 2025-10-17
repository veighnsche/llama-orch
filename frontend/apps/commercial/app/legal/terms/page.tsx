import { TermsPage } from '@rbee/ui/pages'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Terms of Service - rbee',
  description: 'rbee terms of service. Terms and conditions for using rbee services and software.',
  keywords: ['terms of service', 'terms and conditions', 'legal', 'GPL-3.0'],
  alternates: {
    canonical: '/legal/terms',
  },
}

export default function Page() {
  return <TermsPage />
}
