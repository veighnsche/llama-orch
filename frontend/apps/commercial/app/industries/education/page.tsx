import { EducationPage } from '@rbee/ui/pages'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'rbee for Education - Learn Distributed AI Systems',
  description:
    'Learn distributed systems from nature-inspired architecture. Open source (GPL-3.0), BDD-tested. Study real production code.',
  keywords: [
    'learn distributed systems',
    'AI architecture education',
    'open source AI platform',
    'CS education AI',
    'distributed AI tutorial',
    'beehive architecture',
  ],
  alternates: {
    canonical: '/industries/education',
  },
  openGraph: {
    title: 'rbee for Education - Learn Distributed AI',
    description: 'Nature-inspired architecture. Open source, BDD-tested, real production code.',
    type: 'website',
    url: 'https://rbee.dev/industries/education',
    siteName: 'rbee',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'rbee for Education - Distributed AI',
    description: 'Learn distributed systems from nature-inspired architecture.',
  },
}

export default function Page() {
  return <EducationPage />
}
