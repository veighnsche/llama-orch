import { HomelabPage } from '@rbee/ui/pages'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'rbee for Homelab - Self-Hosted AI for Enthusiasts',
  description:
    'Self-hosted AI infrastructure for homelab enthusiasts. SSH-based control, multi-backend support (CUDA, Metal, CPU). Complete control and privacy.',
  keywords: [
    'homelab AI',
    'self-hosted LLM',
    'privacy-first AI',
    'local AI infrastructure',
    'SSH AI control',
    'homelab GPU',
  ],
  alternates: {
    canonical: '/industries/homelab',
  },
  openGraph: {
    title: 'rbee for Homelab - Your Hardware, Your AI',
    description: 'Self-hosted AI for homelab enthusiasts. SSH control, complete privacy, zero external dependencies.',
    type: 'website',
    url: 'https://rbee.dev/industries/homelab',
    siteName: 'rbee',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'rbee for Homelab - Self-Hosted AI',
    description: 'Run AI on your homelab. Complete control, complete privacy.',
  },
}

export default function Page() {
  return <HomelabPage />
}
