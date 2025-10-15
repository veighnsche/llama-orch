import { getPageMap } from 'nextra/page-map'
import { Layout } from 'nextra-theme-docs'

export default async function DocsLayout({ children }: { children: React.ReactNode }) {
  const pageMap = await getPageMap('/docs')

  return (
    <Layout
      pageMap={pageMap}
      docsRepositoryBase="https://github.com/veighnsche/llama-orch/tree/main/frontend/bin/user-docs"
      sidebar={{ defaultMenuCollapseLevel: 1 }}
      footer={<span>{new Date().getFullYear()} © rbee. Private LLM Hosting in the Netherlands.</span>}
    >
      {children}
    </Layout>
  )
}
