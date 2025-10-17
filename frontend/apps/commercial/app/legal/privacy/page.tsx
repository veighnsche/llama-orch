import { PrivacyPage } from '@rbee/ui/pages'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Privacy Policy - rbee',
  description: 'rbee privacy policy. How we collect, use, and protect your data. GDPR-compliant.',
  keywords: ['privacy policy', 'GDPR', 'data protection', 'privacy'],
  alternates: {
    canonical: '/legal/privacy',
  },
}

export default function Page() {
  return <PrivacyPage />
}
