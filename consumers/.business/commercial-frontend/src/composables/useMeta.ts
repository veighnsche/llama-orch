import { onMounted, onBeforeUnmount, watch } from 'vue'

interface MetaOptions {
  title?: string | (() => string)
  description?: string | (() => string)
  keywords?: string[] | (() => string[])
  jsonLd?: Record<string, unknown> | (() => Record<string, unknown>) | null
  jsonLdId?: string
  canonical?: string | (() => string)
  alternates?: Array<{ hrefLang: string; href: string }> | (() => Array<{ hrefLang: string; href: string }>)
  watchSources?: Array<() => unknown> | unknown[]
}

export function useMeta(opts: MetaOptions) {
  const jsonId = opts.jsonLdId ?? 'ld-json-home'
  const canonicalId = 'link-canonical'

  const apply = () => {
    const title = typeof opts.title === 'function' ? opts.title() : opts.title
    if (title) {
      document.title = title
    }

    const description =
      typeof opts.description === 'function' ? opts.description() : opts.description
    if (description) {
      setNamedMeta('description', description)
    }

    const keywords = typeof opts.keywords === 'function' ? opts.keywords() : opts.keywords
    if (keywords && keywords.length) {
      setNamedMeta('keywords', keywords.join(', '))
    }

    // JSON-LD
    removeJsonLd(jsonId)
    const jsonLd = typeof opts.jsonLd === 'function' ? opts.jsonLd() : opts.jsonLd
    if (jsonLd) {
      const script = document.createElement('script')
      script.type = 'application/ld+json'
      script.id = jsonId
      script.text = JSON.stringify(jsonLd)
      document.head.appendChild(script)
    }

    // Canonical
    removeLinkById(canonicalId)
    const canonicalVal = typeof opts.canonical === 'function' ? opts.canonical() : opts.canonical
    if (canonicalVal) {
      const link = document.createElement('link')
      link.setAttribute('rel', 'canonical')
      link.setAttribute('href', canonicalVal)
      link.id = canonicalId
      document.head.appendChild(link)
    }

    // Hreflang alternates
    removeAlternateLinks()
    const alts = typeof opts.alternates === 'function' ? opts.alternates() : opts.alternates
    if (alts && alts.length) {
      for (const a of alts) {
        const link = document.createElement('link')
        link.setAttribute('rel', 'alternate')
        link.setAttribute('hreflang', a.hrefLang)
        link.setAttribute('href', a.href)
        link.setAttribute('data-meta', 'alt')
        document.head.appendChild(link)
      }
    }
  }

  onMounted(apply)
  onBeforeUnmount(() => {
    removeJsonLd(jsonId)
    removeLinkById(canonicalId)
    removeAlternateLinks()
  })

  if (opts.watchSources && opts.watchSources.length) {
    for (const src of opts.watchSources) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      watch(src as any, () => apply())
    }
  }
}

function setNamedMeta(name: string, content: string) {
  let el = document.querySelector(`meta[name="${name}"]`) as HTMLMetaElement | null
  if (!el) {
    el = document.createElement('meta')
    el.setAttribute('name', name)
    document.head.appendChild(el)
  }
  el.setAttribute('content', content)
}

function removeJsonLd(id: string) {
  const prev = document.getElementById(id)
  if (prev) prev.remove()
}

function removeLinkById(id: string) {
  const prev = document.getElementById(id)
  if (prev) prev.remove()
}

function removeAlternateLinks() {
  const links = document.querySelectorAll('link[rel="alternate"][data-meta="alt"]')
  links.forEach((el) => el.remove())
}
