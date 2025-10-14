import { UseCasesHero, UseCasesPrimary, UseCasesIndustry } from '@rbee/ui/organisms/UseCases'
import { EmailCapture } from '@rbee/ui/organisms/EmailCapture'

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
