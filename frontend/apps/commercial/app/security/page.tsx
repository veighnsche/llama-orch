import { SecurityPage } from '@rbee/ui/pages'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'rbee Security - Enterprise-Grade Security Architecture',
  description:
    'Enterprise-grade security architecture. 7-year audit retention, immutable logs, tamper detection. GDPR, SOC2, and ISO 27001 aligned.',
  keywords: ['AI security', 'security architecture', 'audit logs', 'GDPR security', 'SOC2 security', 'ISO 27001'],
  alternates: {
    canonical: '/security',
  },
  openGraph: {
    title: 'rbee Security - Enterprise-Grade Architecture',
    description: '7-year audit retention, immutable logs, tamper detection. SOC2 and ISO 27001 aligned.',
    type: 'website',
    url: 'https://rbee.dev/security',
    siteName: 'rbee',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'rbee Security',
    description: 'Enterprise-grade security with 7-year audit retention.',
  },
}

export default function Page() {
  return <SecurityPage />
}
