import type { Meta, StoryObj } from '@storybook/react'
import { MultiBackendGpu } from './MultiBackendGpu'

const meta = {
	title: 'Organisms/Features/MultiBackendGpu',
	component: MultiBackendGpu,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The MultiBackendGpu section explains rbee's multi-backend GPU support (CUDA, Metal, CPU) and the "GPU FAIL FAST" policy—no silent fallbacks, clear errors with suggestions, explicit backend selection. It demonstrates rbee's philosophy of failing fast with helpful errors rather than silently degrading performance.

## Composition
This organism contains:
- **Section Title**: "Multi-Backend GPU Support"
- **Section Subtitle**: "CUDA, Metal, and CPU backends with explicit device selection"
- **Three Components**:
  1. **GPU FAIL FAST Policy Banner**: Branded banner explaining the policy
  2. **Detection Console**: Terminal showing backend detection results
  3. **Microcards Strip**: 3 feature cards (Detection, Explicit selection, Helpful suggestions)

## Marketing Strategy

### Target Sub-Audience
**Primary**: Technical users who value reliability and clear errors
**Secondary**: Developers frustrated by silent failures in other systems

### Page-Specific Messaging
- **Features page**: Technical deep dive into backend support and fail-fast philosophy
- **Technical level**: Advanced
- **Focus**: Reliability, transparency, control

### Copy Analysis
- **Technical level**: Advanced
- **GPU FAIL FAST Policy**:
  - Prohibited: "No GPU→CPU fallback", "No graceful degradation", "No implicit CPU reroute"
  - What happens: "Fail fast (exit 1)", "Helpful error message", "Explicit backend selection"
  - Example error: "❌ Insufficient VRAM: need 4000 MB, have 2000 MB" with 3 actionable suggestions
- **Detection Console**:
  - Shows backend detection: "cuda × 2", "cpu × 1", "metal × 0"
  - Benefit: "Cached in the registry for fast lookups and policy routing"
- **Microcards**:
  - Detection: "Scans CUDA, Metal, CPU and counts devices"
  - Explicit selection: "Choose backend & device—no surprises"
  - Helpful suggestions: "Actionable fixes on error"

### Conversion Elements
- **Reliability**: "No silent fallbacks" (reduces fear of mystery performance issues)
- **Transparency**: "Clear errors with suggestions" (builds trust)
- **Control**: "You choose the backend" (empowerment)

## When to Use
- On the Features page after IntelligentModelManagement
- To explain multi-backend support
- To demonstrate fail-fast philosophy
- To show error handling quality

## Content Requirements
- **Section Title**: Clear heading
- **Section Subtitle**: Brief overview
- **Policy Banner**: Prohibited behaviors, what happens instead, example error
- **Terminal Example**: Backend detection results
- **Microcards**: Key features (3 cards)

## Variants
- **Default**: Full section with policy banner, console, and microcards
- **CUDA Focus**: Emphasis on CUDA support
- **Metal Focus**: Emphasis on Metal support (Mac users)

## Examples
\`\`\`tsx
import { MultiBackendGpu } from '@rbee/ui/organisms/Features/MultiBackendGpu'

<MultiBackendGpu />
\`\`\`

## Used In
- Features page (\`/features\`)

## Marketing Documentation

### Why "GPU FAIL FAST"?
- **Differentiation**: Most systems silently fall back to CPU, causing mystery performance issues
- **Trust**: Developers trust systems that fail fast with clear errors
- **Control**: Users want to choose their backend explicitly

### Mac Users vs. Linux Users
- **Mac users**: Care about Metal support (M-series chips)
- **Linux users**: Care about CUDA support (NVIDIA GPUs)
- **Unified messaging**: "CUDA, Metal, and CPU" shows support for both

## Technical Implementation
- Uses SectionContainer for consistent layout
- IconBox for feature icons
- Branded policy banner with gradient background
- Terminal-style detection console with faux window chrome

## Related Components
- SectionContainer
- IconBox
- Badge

## Accessibility
- **Keyboard Navigation**: All interactive elements keyboard accessible
- **ARIA Labels**: Detection console has role="region" and aria-label
- **Semantic HTML**: Uses proper heading hierarchy
- **Error Messages**: Error example has role="alert"
        `,
			},
		},
	},
	tags: ['autodocs'],
} satisfies Meta<typeof MultiBackendGpu>

export default meta
type Story = StoryObj<typeof meta>

export const FeaturesPageDefault: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Default multi-backend GPU support section for the Features page. Shows GPU FAIL FAST policy (no silent fallbacks), backend detection console (CUDA, Metal, CPU), and key features (detection, explicit selection, helpful suggestions). Demonstrates fail-fast philosophy.',
			},
		},
	},
}

export const CudaFocus: Story = {
	render: () => (
		<div className="space-y-8">
			<div className="bg-primary/10 p-6 text-center">
				<h3 className="text-xl font-bold">CUDA Support Focus</h3>
				<p className="text-muted-foreground">
					For Linux users with NVIDIA GPUs. Full CUDA support with explicit device selection.
				</p>
			</div>
			<MultiBackendGpu />
			<div className="bg-muted p-8">
				<h3 className="text-xl font-bold mb-4 text-center">CUDA Messaging for Linux Users</h3>
				<div className="max-w-2xl mx-auto space-y-4 text-sm">
					<div className="bg-background p-4 rounded-lg">
						<strong className="block mb-2">Key Benefits for NVIDIA GPU Users:</strong>
						<ul className="space-y-2 text-muted-foreground">
							<li>• Full CUDA support (no compromises)</li>
							<li>• Multi-GPU orchestration (use all your GPUs)</li>
							<li>• Explicit device selection (--device cuda:0, cuda:1)</li>
							<li>• VRAM validation before loading (no crashes)</li>
						</ul>
					</div>
					<div className="bg-background p-4 rounded-lg">
						<strong className="block mb-2">Differentiator:</strong>
						<p className="text-muted-foreground">
							Unlike other systems that silently fall back to CPU when VRAM is insufficient, rbee fails fast with
							clear errors and actionable suggestions.
						</p>
					</div>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Focus on CUDA support for Linux users with NVIDIA GPUs. Emphasizes full CUDA support and fail-fast design.',
			},
		},
	},
}

export const MetalFocus: Story = {
	render: () => (
		<div className="space-y-8">
			<div className="bg-primary/10 p-6 text-center">
				<h3 className="text-xl font-bold">Metal Support Focus</h3>
				<p className="text-muted-foreground">
					For Mac users with M-series chips. Native Metal support for optimal performance.
				</p>
			</div>
			<MultiBackendGpu />
			<div className="bg-muted p-8">
				<h3 className="text-xl font-bold mb-4 text-center">Metal Messaging for Mac Users</h3>
				<div className="max-w-2xl mx-auto space-y-4 text-sm">
					<div className="bg-background p-4 rounded-lg">
						<strong className="block mb-2">Key Benefits for Mac Users:</strong>
						<ul className="space-y-2 text-muted-foreground">
							<li>• Native Metal support (optimized for M-series chips)</li>
							<li>• Unified memory architecture (efficient memory usage)</li>
							<li>• Explicit backend selection (--backend metal)</li>
							<li>• Works alongside CUDA machines in your pool</li>
						</ul>
					</div>
					<div className="bg-background p-4 rounded-lg">
						<strong className="block mb-2">Differentiator:</strong>
						<p className="text-muted-foreground">
							rbee treats Metal as a first-class backend, not an afterthought. Your Mac can work alongside your
							Linux machines seamlessly.
						</p>
					</div>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Focus on Metal support for Mac users with M-series chips. Emphasizes native Metal support and cross-platform orchestration.',
			},
		},
	},
}
