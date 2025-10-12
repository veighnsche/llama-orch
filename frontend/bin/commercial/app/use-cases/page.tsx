import { UseCasesHero } from "@/components/use-cases/use-cases-hero"
import { UseCasesPrimary } from "@/components/use-cases/use-cases-primary"
import { UseCasesIndustry } from "@/components/use-cases/use-cases-industry"
import { EmailCapture } from "@/components/email-capture"

export default function UseCasesPage() {
  return (
    <div className="pt-16">
      <UseCasesHero />
      <UseCasesPrimary />
      <UseCasesIndustry />
      <EmailCapture />
    </div>
  )
}
