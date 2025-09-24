import { onMounted, onBeforeUnmount } from 'vue'

interface MetaOptions {
  title?: string
  description?: string
  keywords?: string[]
  jsonLd?: Record<string, unknown> | null
  jsonLdId?: string
}

export function useMeta(opts: MetaOptions) {
  const jsonId = opts.jsonLdId ?? 'ld-json-home'

  const apply = () => {
    if (opts.title) {
      document.title = opts.title
    }

    if (opts.description) {
      setNamedMeta('description', opts.description)
    }

    if (opts.keywords && opts.keywords.length) {
      setNamedMeta('keywords', opts.keywords.join(', '))
    }

    // JSON-LD
    removeJsonLd(jsonId)
    if (opts.jsonLd) {
      const script = document.createElement('script')
      script.type = 'application/ld+json'
      script.id = jsonId
      script.text = JSON.stringify(opts.jsonLd)
      document.head.appendChild(script)
    }
  }

  onMounted(apply)
  onBeforeUnmount(() => removeJsonLd(jsonId))
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
