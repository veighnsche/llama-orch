import {
  UseCasesHero,
  UseCasesPrimary,
  UseCasesIndustry,
  EmailCapture,
} from '@rbee/ui/organisms'

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
