import type { Meta, StoryObj } from '@storybook/react'
import { ErrorHandling } from './ErrorHandling'

const meta = {
	title: 'Organisms/Features/ErrorHandling',
	component: ErrorHandling,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The ErrorHandling section showcases rbee's comprehensive error handling capabilities—19+ error scenarios across 4 categories (Network, Resource, Model, Process) with clear messages, actionable fixes, and automatic retries. It demonstrates rbee's fail-fast philosophy with helpful error messages rather than silent failures.

## Composition
This organism contains:
- **Section Title**: "Comprehensive Error Handling"
- **Section Subtitle**: "19+ error scenarios with clear messages and actionable fixes"
- **Badge**: "Resiliency"
- **Status KPIs**: 3 metrics (Scenarios covered, Auto-retries, Fail-fast)
- **Error Timeline Console**: Terminal showing retry examples with exponential backoff
- **Playbook Accordion**: 4 expandable categories with error scenarios

## Marketing Strategy

### Target Sub-Audience
**Primary**: Technical users concerned about reliability and operational complexity
**Secondary**: Teams evaluating production readiness

### Page-Specific Messaging
- **Features page**: Technical deep dive into error handling
- **Technical level**: Advanced
- **Focus**: Reliability, transparency, operational excellence

### Copy Analysis
- **Technical level**: Advanced
- **Status KPIs**:
  - "19+ Scenarios covered" (comprehensive)
  - "SSH • HTTP • DL Auto-retries" (automatic recovery)
  - "Clear suggestions Fail-fast" (helpful errors)
- **Error Timeline**:
  - Shows retry logic with exponential backoff and jitter
  - Example: "[ssh] attempt 1 → timeout (5000ms)" → "retry in 0.8× backoff (1.2s jitter)"
  - Demonstrates helpful suggestions: "check ~/.ssh/config or key permissions"
- **Playbook Categories**:
  1. **Network & Connectivity**: SSH timeouts, auth failures, HTTP errors (4 scenarios)
  2. **Resource Errors**: RAM/VRAM limits, disk space, OOM (4 scenarios)
  3. **Model & Backend**: Model 404, private model 403, download failures (4 scenarios)
  4. **Process Lifecycle**: Worker crashes, graceful shutdown, force-kill (4 scenarios)

### Conversion Elements
- **Reliability**: "19+ scenarios" (comprehensive coverage)
- **Transparency**: "Clear messages and actionable fixes" (no mystery failures)
- **Automation**: "Auto-retries" (reduces operational burden)
- **Developer experience**: Helpful error messages reduce debugging time

## When to Use
- On the Features page after MultiBackendGpu
- To demonstrate error handling quality
- To show operational maturity
- To build confidence in production readiness

## Content Requirements
- **Section Title**: Clear heading
- **Section Subtitle**: Brief overview
- **Badge**: Feature category
- **Status KPIs**: 3 key metrics
- **Error Timeline**: Terminal with retry examples
- **Playbook**: 4 expandable categories with scenarios

## Variants
- **Default**: Full section with all categories
- **Cascading Shutdown Focus**: Emphasis on clean teardown
- **VRAM Cleanup Focus**: Emphasis on memory management

## Examples
\`\`\`tsx
import { ErrorHandling } from '@rbee/ui/organisms/Features/ErrorHandling'

<ErrorHandling />
\`\`\`

## Used In
- Features page (\`/features\`)

## Marketing Documentation

### Why "19+ Scenarios"?
- **Specificity**: "19+" is more credible than "comprehensive" or "many"
- **Confidence**: Shows the team has thought through edge cases
- **Differentiation**: Most systems have poor error handling

### Developer Pain Point
Developers hate:
- Silent failures (GPU→CPU fallback without warning)
- Cryptic error messages ("Error 500")
- No actionable fixes ("Something went wrong")

rbee addresses all three:
- Fail-fast (no silent fallbacks)
- Clear messages ("Insufficient VRAM: need 4000 MB, have 2000 MB")
- Actionable fixes ("Use smaller quantized model (Q4_K_M instead of Q8_0)")

## Technical Implementation
- Uses SectionContainer for consistent layout
- PlaybookAccordion for expandable categories
- TerminalConsole for error timeline
- StatusKPI for metrics
- Interactive expand/collapse all buttons

## Related Components
- SectionContainer
- PlaybookAccordion
- TerminalConsole
- StatusKPI

## Accessibility
- **Keyboard Navigation**: Accordion items keyboard accessible
- **ARIA Labels**: Terminal has aria-label and aria-live="polite"
- **Semantic HTML**: Uses \`<details>\` for accordion
- **Focus Management**: Focus moves to opened accordion item
        `,
			},
		},
	},
	tags: ['autodocs'],
} satisfies Meta<typeof ErrorHandling>

export default meta
type Story = StoryObj<typeof meta>

export const FeaturesPageDefault: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Default error handling section for the Features page. Shows 19+ error scenarios across 4 categories (Network, Resource, Model, Process) with clear messages, actionable fixes, and automatic retries. Includes error timeline with retry logic and expandable playbook.',
			},
		},
	},
}

export const CascadingShutdownFocus: Story = {
	render: () => (
		<div className="space-y-8">
			<div className="bg-primary/10 p-6 text-center">
				<h3 className="text-xl font-bold">Cascading Shutdown Focus</h3>
				<p className="text-muted-foreground">
					Ctrl+C cleanly tears down keeper → queen → hive → workers. No orphans, no VRAM leaks.
				</p>
			</div>
			<ErrorHandling />
			<div className="bg-muted p-8">
				<h3 className="text-xl font-bold mb-4 text-center">Pain Point Addressed</h3>
				<div className="max-w-2xl mx-auto space-y-4 text-sm">
					<div className="bg-background p-4 rounded-lg">
						<strong className="block mb-2">Without rbee (typical distributed systems):</strong>
						<ul className="space-y-1 text-muted-foreground">
							<li>• Ctrl+C kills orchestrator, leaves workers running</li>
							<li>• Orphaned processes consume VRAM indefinitely</li>
							<li>• Manual cleanup required (kill -9, nvidia-smi)</li>
							<li>• Mystery VRAM leaks accumulate over time</li>
						</ul>
					</div>
					<div className="bg-background p-4 rounded-lg">
						<strong className="block mb-2">With rbee:</strong>
						<ul className="space-y-1 text-muted-foreground">
							<li>• Ctrl+C triggers cascading shutdown</li>
							<li>• Workers receive shutdown signal</li>
							<li>• VRAM cleaned up automatically</li>
							<li>• No orphaned processes, no leaks</li>
						</ul>
					</div>
					<div className="bg-background p-4 rounded-lg">
						<strong className="block mb-2">Key Differentiator:</strong>
						<p className="text-muted-foreground">
							"Graceful shutdown with active requests" + "Force-kill after 30s timeout" shows rbee handles both
							the happy path (clean shutdown) and the failure path (force-kill if needed).
						</p>
					</div>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story:
					'Focus on cascading shutdown capabilities. Compares typical distributed systems (orphaned processes) vs. rbee (clean teardown).',
			},
		},
	},
}

export const VramCleanupFocus: Story = {
	render: () => (
		<div className="space-y-8">
			<ErrorHandling />
			<div className="bg-muted p-8">
				<h3 className="text-xl font-bold mb-4 text-center">VRAM Cleanup Strategy</h3>
				<div className="max-w-3xl mx-auto space-y-4 text-sm">
					<p className="text-muted-foreground">
						rbee's VRAM management is a key differentiator for GPU users:
					</p>
					<div className="grid md:grid-cols-2 gap-4">
						<div className="bg-background p-4 rounded-lg">
							<strong className="block mb-2">Preflight Checks</strong>
							<p className="text-muted-foreground">
								Before loading: "Requires available RAM ≥ model size × 1.2" and "Sufficient GPU VRAM for selected
								backend"
							</p>
						</div>
						<div className="bg-background p-4 rounded-lg">
							<strong className="block mb-2">Fail-Fast on VRAM</strong>
							<p className="text-muted-foreground">
								"VRAM exhausted" → "no CPU fallback" (explicit, no silent degradation)
							</p>
						</div>
						<div className="bg-background p-4 rounded-lg">
							<strong className="block mb-2">Clean Shutdown</strong>
							<p className="text-muted-foreground">
								"VRAM cleanup" on shutdown (no leaks, no orphaned allocations)
							</p>
						</div>
						<div className="bg-background p-4 rounded-lg">
							<strong className="block mb-2">OOM Handling</strong>
							<p className="text-muted-foreground">
								"OOM during model load" → "abort safely" (no crash, clean error)
							</p>
						</div>
					</div>
					<div className="bg-background p-4 rounded-lg">
						<strong className="block mb-2">Why This Matters:</strong>
						<p className="text-muted-foreground">
							GPU users are frustrated by mystery VRAM leaks and silent fallbacks. rbee's explicit VRAM management
							and fail-fast policy build trust.
						</p>
					</div>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story:
					'Focus on VRAM cleanup and management. Shows preflight checks, fail-fast policy, clean shutdown, and OOM handling.',
			},
		},
	},
}
