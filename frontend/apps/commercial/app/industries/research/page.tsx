import { ResearchPage } from '@rbee/ui/pages'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'rbee for Research - Reproducible ML Experiments',
  description:
    'Multi-modal AI platform for researchers. Proof bundles for reproducibility, BDD-tested, determinism suite. Research-grade quality with production infrastructure.',
  keywords: [
    'AI research infrastructure',
    'reproducible ML',
    'multi-modal AI platform',
    'research AI tools',
    'ML experiment tracking',
    'proof bundles',
  ],
  alternates: {
    canonical: '/industries/research',
  },
  openGraph: {
    title: 'rbee for Research - Reproducible AI Experiments',
    description: 'Multi-modal AI platform. Proof bundles, determinism suite, research-grade quality.',
    type: 'website',
    url: 'https://rbee.dev/industries/research',
    siteName: 'rbee',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'rbee for Research - Reproducible ML',
    description: 'Research-grade AI infrastructure with proof bundles and determinism.',
  },
}

export default function Page() {
  return <ResearchPage />
}
