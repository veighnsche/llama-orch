import { ProblemSection } from '@rbee/ui/organisms'
import { AlertTriangle, FileX, Globe, Scale } from 'lucide-react'

/**
 * Backward-compatible wrapper for the enterprise page.
 * Re-exports the shared ProblemSection with enterprise-specific defaults.
 * Note: Uses 4-column grid (extended from default 3).
 */
export function EnterpriseProblem() {
	return (
		<ProblemSection
			kicker="The Compliance Risk"
			title="The Compliance Challenge of Cloud AI"
			subtitle="Using external AI providers creates compliance risks that can cost millions in fines and damage your reputation."
			items={[
				{
					icon: <Globe className="h-6 w-6" />,
					title: 'Data Sovereignty Violations',
					body: 'Your sensitive data crosses borders to US cloud providers. GDPR Article 44 violations. Schrems II compliance impossible. Data Protection Authorities watching.',
					tone: 'destructive',
					tag: 'GDPR Art. 44',
				},
				{
					icon: <FileX className="h-6 w-6" />,
					title: 'Missing Audit Trails',
					body: 'No immutable logs. No proof of compliance. Cannot demonstrate GDPR Article 30 compliance. SOC2 audits fail. ISO 27001 certification impossible.',
					tone: 'destructive',
					tag: 'Audit failure',
				},
				{
					icon: <Scale className="h-6 w-6" />,
					title: 'Regulatory Fines',
					body: 'GDPR fines up to €20M or 4% of global revenue. Healthcare (HIPAA) violations: $50K per record. Financial services (PCI-DSS) breaches: reputation destroyed.',
					tone: 'destructive',
					tag: 'Up to €20M',
				},
				{
					icon: <AlertTriangle className="h-6 w-6" />,
					title: 'Zero Control',
					body: 'Provider changes terms. Data Processing Agreements worthless. Cannot guarantee data residency. Cannot prove compliance. Your DPO cannot sleep.',
					tone: 'destructive',
					tag: 'No guarantees',
				},
			]}
			ctaPrimary={{ label: 'Request Demo', href: '/enterprise/demo' }}
			ctaSecondary={{ label: 'Compliance Overview', href: '/enterprise/compliance' }}
			ctaCopy='"We cannot use external AI providers due to GDPR compliance requirements." — Every EU CTO and Data Protection Officer'
			gridClassName="md:grid-cols-2 lg:grid-cols-4"
		/>
	)
}
