import { FeatureTabsSection } from '@rbee/ui/organisms'
import { Code, Cpu, Gauge, Terminal } from 'lucide-react'

export function DevelopersFeatures() {
	return (
		<FeatureTabsSection
			id="features"
			title="Enterprise-grade features. Homelab simplicity."
			items={[
				{
					id: 'openai-api',
					title: 'OpenAI-Compatible API',
					description: 'Drop-in for OpenAI. Works with Zed, Cursor, Continue — any OpenAI client.',
					icon: <Code className="h-4 w-4" />,
					benefit: { text: 'No code changes. Just point to localhost.' },
					example: {
						kind: 'terminal',
						title: 'Example',
						content: `export OPENAI_API_BASE=http://localhost:8080/v1
export OPENAI_API_KEY=your-rbee-token

# Now Zed IDE uses YOUR infrastructure
zed .`,
						copyText: 'export OPENAI_API_BASE=http://localhost:8080/v1\nexport OPENAI_API_KEY=your-rbee-token\nzed .',
					},
				},
				{
					id: 'multi-gpu',
					title: 'Multi-GPU Orchestration',
					description: 'Distribute across CUDA, Metal, and CPU backends. Use every GPU you own.',
					icon: <Cpu className="h-4 w-4" />,
					benefit: { text: '10× throughput by using all your hardware.' },
					example: {
						kind: 'terminal',
						title: 'Example',
						content: `rbee-keeper worker start --gpu 0 --backend cuda  # PC
rbee-keeper worker start --gpu 1 --backend cuda  # PC
rbee-keeper worker start --gpu 0 --backend metal # Mac

# All GPUs work together automatically`,
						copyText:
							'rbee-keeper worker start --gpu 0 --backend cuda\nrbee-keeper worker start --gpu 1 --backend cuda\nrbee-keeper worker start --gpu 0 --backend metal',
					},
				},
				{
					id: 'task-api',
					title: 'Task-Based API with SSE',
					description: 'See model loading, token generation, and cost tracking in real time.',
					icon: <Terminal className="h-4 w-4" />,
					benefit: { text: 'Full visibility into every inference job.' },
					example: {
						kind: 'terminal',
						title: 'SSE Stream',
						content: `event: started
data: {"queue_position":3}

event: token
data: {"t":"Hello","i":0}

event: metrics
data: {"tokens_remaining":98}`,
					},
				},
				{
					id: 'shutdown',
					title: 'Cascading Shutdown',
					description: 'Press Ctrl+C once. Everything stops cleanly. No orphaned processes or leaked VRAM.',
					icon: <Gauge className="h-4 w-4" />,
					benefit: { text: 'Reliable cleanup. Safe for development.' },
					example: {
						kind: 'terminal',
						title: 'Example',
						content: `# Press Ctrl+C
^C
Shutting down queen-rbee...
Stopping all hives...
Terminating workers...
✓ Clean shutdown complete`,
					},
				},
			]}
		/>
	)
}
