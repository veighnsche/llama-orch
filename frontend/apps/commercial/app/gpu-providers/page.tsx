import {
	EmailCapture,
	Footer,
	ProvidersCTA,
	ProvidersEarnings,
	ProvidersFeatures,
	ProvidersHero,
	ProvidersHowItWorks,
	ProvidersMarketplace,
	ProvidersProblem,
	ProvidersSecurity,
	ProvidersSolution,
	ProvidersTestimonials,
	ProvidersUseCases,
} from '@rbee/ui/organisms'

export default function GPUProvidersPage() {
	return (
		<main className="min-h-screen bg-background">
			<ProvidersHero />
			<ProvidersProblem />
			<ProvidersSolution />
			<ProvidersHowItWorks />
			<ProvidersFeatures />
			<ProvidersUseCases />
			<ProvidersEarnings />
			<ProvidersMarketplace />
			<ProvidersSecurity />
			<ProvidersTestimonials />
			<ProvidersCTA />
		</main>
	)
}
