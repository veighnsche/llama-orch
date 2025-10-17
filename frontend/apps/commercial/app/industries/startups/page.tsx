import { StartupsPage } from '@rbee/ui/pages'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'rbee for Startups - Scale Your AI Infrastructure',
  description:
    'Build your AI infrastructure to escape dependency on big AI providers. Year 1: 35 customers, €70K. Independence from external providers.',
  keywords: [
    'AI startup infrastructure',
    'OpenAI alternative for startups',
    'self-hosted AI business',
    'AI provider independence',
    'startup AI platform',
    'scale AI infrastructure',
  ],
  alternates: {
    canonical: '/industries/startups',
  },
  openGraph: {
    title: 'rbee for Startups - Scale Your AI Infrastructure',
    description: 'Build your own AI infrastructure. Year 1: 35 customers, €70K revenue. Independence from providers.',
    type: 'website',
    url: 'https://rbee.dev/industries/startups',
    siteName: 'rbee',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'rbee for Startups - AI Independence',
    description: 'Build AI infrastructure on your terms. Escape provider dependency.',
  },
}

export default function Page() {
  return <StartupsPage />
}
