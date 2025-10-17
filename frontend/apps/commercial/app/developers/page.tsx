import { DevelopersPage } from '@rbee/ui/pages'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'rbee for Developers - Build AI Tools Without Vendor Lock-In',
  description:
    'OpenAI-compatible API for developers. Build AI coders, doc generators, test creators on your hardware. Zero API fees, complete privacy. Works with Zed & Cursor.',
  keywords: [
    'developer AI tools',
    'OpenAI compatible API',
    'local AI development',
    'AI coding assistant',
    'self-hosted AI for developers',
    'Cursor AI',
    'Zed AI',
    'privacy-first AI',
  ],
  alternates: {
    canonical: '/developers',
  },
  openGraph: {
    title: 'rbee for Developers - Build AI Tools on Your Hardware',
    description:
      'OpenAI-compatible API. Build AI coders, doc generators, test creators. Zero vendor lock-in, complete control.',
    type: 'website',
    url: 'https://rbee.dev/developers',
    siteName: 'rbee',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'rbee for Developers - No Vendor Lock-In',
    description: 'Build AI tools on your hardware. OpenAI-compatible API, zero fees, complete privacy.',
  },
}

export default function Developers() {
  return <DevelopersPage />
}
