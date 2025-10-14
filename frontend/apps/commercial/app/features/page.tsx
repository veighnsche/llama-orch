import {
	AdditionalFeaturesGrid,
	CoreFeaturesTabs,
	CrossNodeOrchestration,
	EmailCapture,
	ErrorHandling,
	FeaturesHero,
	IntelligentModelManagement,
	MultiBackendGpu,
	RealTimeProgress,
	SecurityIsolation,
} from '@rbee/ui/organisms'

export default function FeaturesPage() {
	return (
		<>
			<FeaturesHero />
			<CoreFeaturesTabs />
			<CrossNodeOrchestration />
			<IntelligentModelManagement />
			<MultiBackendGpu />
			<ErrorHandling />
			<RealTimeProgress />
			<SecurityIsolation />
			<AdditionalFeaturesGrid />
			<EmailCapture />
		</>
	)
}
