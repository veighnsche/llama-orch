import type { Language } from 'prism-react-renderer'

/**
 * Resolve language string to Prism Language type.
 * Defaults to 'tsx' for unknown languages.
 */
export const resolveLang = (lang?: string): Language => {
  if (!lang) return 'tsx'

  // Map common aliases to Prism languages
  const langMap: Record<string, Language> = {
    typescript: 'tsx',
    ts: 'tsx',
    javascript: 'jsx',
    js: 'jsx',
    python: 'python',
    py: 'python',
    bash: 'bash',
    sh: 'bash',
    shell: 'bash',
    json: 'json',
    yaml: 'yaml',
    yml: 'yaml',
    markdown: 'markdown',
    md: 'markdown',
    css: 'css',
    html: 'markup',
    xml: 'markup',
    rust: 'rust',
    rs: 'rust',
    go: 'go',
    sql: 'sql',
  }

  const normalized = lang.toLowerCase()
  return (langMap[normalized] as Language) || (lang as Language)
}
