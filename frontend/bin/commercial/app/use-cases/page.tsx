import { UseCasesHero } from '@/components/organisms/UseCases/use-cases-hero'
import { UseCasesPrimary } from '@/components/organisms/UseCases/use-cases-primary'
import { UseCasesIndustry } from '@/components/organisms/UseCases/use-cases-industry'
import { EmailCapture } from '@/components/organisms/EmailCapture/EmailCapture'

export default function UseCasesPage() {
  return (
    <>
      <UseCasesHero />
      <div id="use-cases">
        <UseCasesPrimary />
      </div>
      <div id="architecture">
        <UseCasesIndustry />
      </div>
      <EmailCapture />
    </>
  )
}
