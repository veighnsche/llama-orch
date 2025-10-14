import { Button } from '@rbee/ui/atoms/Button'
import { CompliancePillar } from '@rbee/ui/molecules/CompliancePillar'
import { Globe, Lock, Shield } from 'lucide-react'
import Image from 'next/image'
import Link from 'next/link'

export function EnterpriseCompliance() {
	return (
		<section
			id="compliance"
			aria-labelledby="compliance-h2"
			role="region"
			className="relative border-b border-border bg-radial-glow px-6 py-24"
		>
			{/* Decorative background illustration */}
			<Image
				src="/decor/compliance-ledger.webp"
				width={1200}
				height={640}
				className="pointer-events-none absolute left-1/2 top-6 -z-10 hidden w-[50rem] -translate-x-1/2 opacity-15 blur-[0.5px] md:block"
				alt="Abstract EU-blue ledger lines with checkpoint nodes; evokes immutable audit trails, GDPR alignment, SOC2 controls, ISO 27001 ISMS"
				aria-hidden="true"
			/>

			<div className="relative z-10 mx-auto max-w-7xl">
				{/* Header */}
				<div className="animate-in fade-in-50 slide-in-from-bottom-2 mb-16 text-center duration-500">
					<p className="mb-2 text-sm font-medium text-primary/80">Security & Certifications</p>
					<h2 id="compliance-h2" className="mb-4 text-4xl font-bold text-foreground">
						Compliance by Design
					</h2>
					<p className="mx-auto max-w-3xl text-balance text-xl text-foreground/85">
						Built from the ground up to meet GDPR, SOC2, and ISO 27001 requirements—security is engineered in, not
						bolted on.
					</p>
				</div>

				{/* Three Pillars */}
				<div className="animate-in fade-in-50 mb-12 grid gap-8 [animation-delay:120ms] lg:grid-cols-3">
					{/* GDPR Pillar */}
					<CompliancePillar
						icon={<Globe className="h-6 w-6" aria-hidden="true" />}
						title="GDPR"
						subtitle="EU Regulation"
						checklist={[
							'7-year audit retention (Art. 30)',
							'Data access records (Art. 15)',
							'Erasure tracking (Art. 17)',
							'Consent management (Art. 7)',
							'Data residency controls (Art. 44)',
							'Breach notification (Art. 33)',
						]}
						callout={
							<div className="rounded-lg border border-chart-3/50 bg-chart-3/10 p-4">
								<div className="mb-2 font-semibold text-chart-3">Compliance Endpoints</div>
								<div className="space-y-1 font-mono text-xs text-foreground/85">
									<div>GET /v2/compliance/data-access</div>
									<div>POST /v2/compliance/data-export</div>
									<div>POST /v2/compliance/data-deletion</div>
									<div>GET /v2/compliance/audit-trail</div>
								</div>
							</div>
						}
					/>

					{/* SOC2 Pillar */}
					<CompliancePillar
						icon={<Shield className="h-6 w-6" aria-hidden="true" />}
						title="SOC2"
						subtitle="US Standard"
						checklist={[
							'Auditor query API',
							'32 audit event types',
							'7-year retention (Type II)',
							'Tamper-evident hash chains',
							'Access control logging',
							'Encryption at rest',
						]}
						callout={
							<div className="rounded-lg border border-chart-3/50 bg-chart-3/10 p-4">
								<div className="mb-2 font-semibold text-chart-3">Trust Service Criteria</div>
								<div className="space-y-1 text-xs text-foreground/85">
									<div>✓ Security (CC1-CC9)</div>
									<div>✓ Availability (A1.1-A1.3)</div>
									<div>✓ Confidentiality (C1.1-C1.2)</div>
								</div>
							</div>
						}
					/>

					{/* ISO 27001 Pillar */}
					<CompliancePillar
						icon={<Lock className="h-6 w-6" aria-hidden="true" />}
						title="ISO 27001"
						subtitle="International Standard"
						checklist={[
							'Incident records (A.16)',
							'3-year minimum retention',
							'Access logging (A.9)',
							'Crypto controls (A.10)',
							'Ops security (A.12)',
							'Security policies (A.5)',
						]}
						callout={
							<div className="rounded-lg border border-chart-3/50 bg-chart-3/10 p-4">
								<div className="mb-2 font-semibold text-chart-3">ISMS Controls</div>
								<div className="space-y-1 text-xs text-foreground/85">
									<div>✓ 114 controls implemented</div>
									<div>✓ Risk assessment framework</div>
									<div>✓ Continuous monitoring</div>
								</div>
							</div>
						}
					/>
				</div>

				{/* Audit Readiness Band */}
				<div className="animate-in fade-in-50 rounded-2xl border border-primary/20 bg-primary/5 p-8 text-center [animation-delay:200ms]">
					<h3 className="mb-2 text-2xl font-semibold text-foreground">Ready for Your Compliance Audit</h3>
					<p className="mb-2 text-foreground/85">
						Download our compliance documentation package or schedule a call with our compliance team.
					</p>
					<p
						id="compliance-pack-note"
						className="mb-6 text-sm text-muted-foreground"
						aria-label="Compliance pack includes endpoints, retention policy, and audit-logging design"
					>
						Pack includes endpoints, retention policy, and audit-logging design.
					</p>
					<div className="flex flex-col items-center justify-center gap-3 sm:flex-row">
						<Button
							size="lg"
							asChild
							aria-describedby="compliance-pack-note"
							className="transition-transform active:scale-[0.98]"
						>
							<Link href="/compliance/download">Download Compliance Pack</Link>
						</Button>
						<Button
							size="lg"
							variant="outline"
							asChild
							aria-describedby="compliance-pack-note"
							className="transition-transform active:scale-[0.98]"
						>
							<Link href="/contact/compliance">Talk to Compliance Team</Link>
						</Button>
					</div>
					<p className="mt-6 text-xs text-muted-foreground">rbee (pronounced "are-bee")</p>
				</div>
			</div>
		</section>
	)
}
