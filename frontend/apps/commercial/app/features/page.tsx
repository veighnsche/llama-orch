import { FeaturesPage } from '@rbee/ui/pages'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'rbee Features - Multi-GPU Orchestration, Error Handling & Security',
  description:
    'Comprehensive rbee features: cross-node orchestration, intelligent model management, multi-backend GPU support, 19+ error scenarios, enterprise-grade security. OpenAI-compatible.',
  keywords: [
    'multi-GPU orchestration',
    'error handling',
    'real-time progress',
    'security isolation',
    'model management',
    'GPU support',
    'CUDA',
    'Metal',
    'OpenAI compatible',
  ],
  alternates: {
    canonical: '/features',
  },
  openGraph: {
    title: 'rbee Features - Enterprise-Grade AI Infrastructure',
    description:
      'Multi-GPU orchestration, intelligent error handling, real-time progress tracking. Built for production workloads.',
    type: 'website',
    url: 'https://rbee.dev/features',
    siteName: 'rbee',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'rbee Features - Multi-GPU Orchestration & Security',
    description: 'Cross-node orchestration, 19+ error scenarios, enterprise-grade security. OpenAI-compatible API.',
  },
}

export default function Page() {
  return <FeaturesPage />
}
