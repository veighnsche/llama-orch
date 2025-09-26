<script setup lang="ts">
  import { computed } from 'vue'
  import { useI18n } from 'vue-i18n'

  const { t } = useI18n()

  const props = withDefaults(
    defineProps<{
      compact?: boolean
      appendUtm?: boolean
    }>(),
    { compact: false, appendUtm: false },
  )

  const env = import.meta.env
  const docsUrl = env.VITE_DOCS_URL as string | undefined
  const apiUrl = env.VITE_API_REF_URL as string | undefined
  const githubUrl = env.VITE_GITHUB_URL as string | undefined

  type LinkKey = 'docs' | 'api' | 'github'

  const links = computed(() => {
    const items: Array<{ key: LinkKey; href: string; external: boolean }> = []
    if (docsUrl && /^https?:\/\//.test(docsUrl)) items.push({ key: 'docs', href: docsUrl, external: true })
    else if (githubUrl && /^https?:\/\//.test(githubUrl)) items.push({ key: 'docs', href: `${githubUrl}#readme`, external: true })
    if (apiUrl && /^https?:\/\//.test(apiUrl)) items.push({ key: 'api', href: apiUrl, external: true })
    if (githubUrl && /^https?:\/\//.test(githubUrl)) items.push({ key: 'github', href: githubUrl, external: true })
    return items
  })

  // Final links with optional UTM params appended for external links
  const finalLinks = computed(() => {
    if (!props.appendUtm) return links.value
    const params: Record<string, string> = {
      utm_source: 'site',
      utm_medium: 'devbar',
      utm_campaign: 'homepage',
    }
    return links.value.map((l) => {
      if (!l.external) return l
      try {
        const u = new URL(l.href)
        for (const [k, v] of Object.entries(params)) {
          if (!u.searchParams.has(k)) u.searchParams.set(k, v)
        }
        return { ...l, href: u.toString() }
      } catch {
        return l
      }
    })
  })
</script>

<template>
  <nav
    class="devbar"
    :class="{ compact: props.compact }"
    :aria-label="t('devbar.aria', 'Developer shortcuts')"
    v-if="finalLinks.length"
  >
    <div class="container">
      <div class="left">
        <span class="label">{{ t('devbar.label', 'Developer') }}</span>
        <ul class="links">
          <li v-for="(l, i) in finalLinks" :key="i">
            <a :href="l.href" target="_blank" rel="noopener noreferrer">
              {{ t(`devbar.${l.key}`) }}
            </a>
          </li>
        </ul>
      </div>
    </div>
  </nav>
</template>

<style scoped>
  .devbar {
    background: var(--surface-alt);
    color: var(--muted);
    border-bottom: 1px solid var(--surface-muted);
    font-size: 0.875rem;
  }
  .container {
    max-width: 1120px;
    margin: 0 auto;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.35rem 1rem; /* slim */
  }
  .left {
    display: inline-flex;
    align-items: center;
    gap: 0.75rem;
    min-width: 0; /* keep compact */
  }
  .label {
    color: var(--muted);
    font-weight: 700;
    letter-spacing: .02em;
    text-transform: uppercase;
    font-size: 0.75rem;
    opacity: 0.85;
  }
  .links {
    display: inline-flex;
    gap: 0.75rem;
    list-style: none;
    padding: 0;
    margin: 0;
  }
  a {
    color: var(--text);
    text-decoration: none;
    font-weight: 600;
  }
  a:hover,
  a:focus {
    text-decoration: underline;
    outline: none;
  }
  /* Compact mode: reduce horizontal padding and gaps */
  .devbar.compact .container {
    padding: 0.2rem 0.5rem;
    gap: 0.4rem;
  }
  .devbar.compact .left,
  .devbar.compact .links {
    gap: 0.4rem;
  }
  /* Manual dark override */
  :root[data-theme='dark'] .devbar,
  .dark .devbar {
    background: color-mix(in srgb, var(--surface) 85%, transparent);
  }
</style>
