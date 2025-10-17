import { PricingPage } from '@rbee/ui/pages'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'rbee Pricing - Start Free, Scale When Ready | €0 to €99/mo',
  description:
    'Transparent rbee pricing: Home/Lab (€0 forever), Team (€99/mo), Enterprise (custom). No hidden fees, no per-token costs. Self-hosted AI infrastructure on your terms.',
  keywords: [
    'rbee pricing',
    'free AI hosting',
    'self-hosted pricing',
    'OpenAI alternative cost',
    'GPU orchestration pricing',
    'enterprise AI pricing',
  ],
  alternates: {
    canonical: '/pricing',
  },
  openGraph: {
    title: 'rbee Pricing - Start Free, Scale When Ready',
    description: 'Home/Lab: €0 forever. Team: €99/mo. Enterprise: Custom. No per-token costs, complete control.',
    type: 'website',
    url: 'https://rbee.dev/pricing',
    siteName: 'rbee',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'rbee Pricing - €0 to €99/mo',
    description: 'Start free, scale when ready. No hidden fees, no per-token costs. Own your AI infrastructure.',
  },
}

export default function Pricing() {
  return <PricingPage />
}
