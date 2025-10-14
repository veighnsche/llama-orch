import {
	EmailCapture,
	EnterpriseComparison,
	EnterpriseCompliance,
	EnterpriseCTA,
	EnterpriseFeatures,
	EnterpriseHero,
	EnterpriseHowItWorks,
	EnterpriseProblem,
	EnterpriseSecurity,
	EnterpriseSolution,
	EnterpriseTestimonials,
	EnterpriseUseCases,
	Footer,
} from '@rbee/ui/organisms'

export default function EnterprisePage() {
	return (
		<main className="min-h-screen bg-slate-950">
			<EnterpriseHero />
			<EmailCapture />
			<EnterpriseProblem />
			<EnterpriseSolution />
			<EnterpriseCompliance />
			<EnterpriseSecurity />
			<EnterpriseHowItWorks />
			<EnterpriseUseCases />
			<EnterpriseComparison />
			<EnterpriseFeatures />
			<EnterpriseTestimonials />
			<EnterpriseCTA />
		</main>
	)
}
