import { UseCasesPage } from '@rbee/ui/pages'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'rbee Use Cases - AI Infrastructure for Developers, Teams & Enterprise',
  description:
    'Discover how developers, startups, enterprises, and researchers use rbee for private AI infrastructure. From solo projects to GDPR-compliant enterprise deployments.',
  keywords: [
    'AI use cases',
    'self-hosted AI examples',
    'developer AI',
    'enterprise AI use cases',
    'team AI infrastructure',
    'research AI',
    'homelab AI',
    'private AI applications',
  ],
  alternates: {
    canonical: '/use-cases',
  },
  openGraph: {
    title: 'rbee Use Cases - From Solo Devs to Enterprise',
    description:
      'See how developers, teams, and enterprises use rbee. Solo projects, team collaboration, GDPR-compliant deployments.',
    type: 'website',
    url: 'https://rbee.dev/use-cases',
    siteName: 'rbee',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'rbee Use Cases - Developers to Enterprise',
    description: 'Private AI infrastructure for solo devs, teams, enterprises, and researchers.',
  },
}

export default function UseCases() {
  return <UseCasesPage />
}
