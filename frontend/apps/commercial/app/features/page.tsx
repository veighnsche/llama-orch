import { Metadata } from 'next'
import { FeaturesPage } from '@rbee/ui/pages'

export const metadata: Metadata = {
  title: 'Features | rbee',
  description:
    'Explore rbee features: cross-node orchestration, intelligent model management, multi-backend GPU support, comprehensive error handling, real-time progress tracking, and security isolation.',
}

export default function Page() {
  return <FeaturesPage />
}
