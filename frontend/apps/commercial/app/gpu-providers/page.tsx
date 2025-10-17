import { ProvidersPage } from '@rbee/ui/pages'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Earn with Your GPU - rbee Marketplace | €50-200/mo Passive Income',
  description:
    'Monetize idle GPUs on the rbee marketplace. Set your own pricing, 85% payout rate, weekly payments. Join 500+ providers earning passive income from spare compute.',
  keywords: [
    'GPU rental',
    'earn with GPU',
    'passive income GPU',
    'GPU marketplace',
    'monetize GPU',
    'idle GPU income',
    'NVIDIA rental',
    'GPU hosting',
  ],
  alternates: {
    canonical: '/gpu-providers',
  },
  openGraph: {
    title: 'Earn €50-200/mo with Your Idle GPU',
    description:
      'Turn idle GPUs into monthly income. Set your pricing, 85% payout, weekly payments. Join 500+ providers.',
    type: 'website',
    url: 'https://rbee.dev/gpu-providers',
    siteName: 'rbee',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Earn with Your GPU - rbee Marketplace',
    description: 'Monetize idle GPUs. €50-200/mo passive income, 85% payout rate, weekly payments.',
  },
}

export default function GPUProviders() {
  return <ProvidersPage />
}
