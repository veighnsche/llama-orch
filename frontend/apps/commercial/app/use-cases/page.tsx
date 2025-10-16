import {
  EmailCapture,
  UseCasesHero,
  UseCasesIndustry,
  UseCasesPrimary,
} from "@rbee/ui/organisms";

export default function UseCasesPage() {
  return (
    <>
      <UseCasesHero />
      <UseCasesPrimary />
      <UseCasesIndustry />
      <EmailCapture />
    </>
  );
}
