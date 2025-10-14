import { Navbar } from 'nextra-theme-docs'

export function CustomNavbar() {
	return (
		<Navbar>
			<div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
				<span style={{ fontWeight: 700 }}>rbee Docs</span>
				<a
					href="https://github.com/veighnsche/llama-orch"
					target="_blank"
					rel="noopener noreferrer"
					style={{ marginLeft: 'auto' }}
				>
					GitHub
				</a>
			</div>
		</Navbar>
	)
}
