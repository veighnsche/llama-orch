<script setup lang="ts">
  import { computed } from 'vue'

  const env = import.meta.env
  const docsUrl = env.VITE_DOCS_URL as string | undefined
  const apiUrl = env.VITE_API_REF_URL as string | undefined
  const githubUrl = env.VITE_GITHUB_URL as string | undefined

  const links = computed(() => {
    const items: Array<{ label: string; href: string; external: boolean }> = []
    if (docsUrl && /^https?:\/\//.test(docsUrl)) items.push({ label: 'Docs', href: docsUrl, external: true })
    else if (githubUrl && /^https?:\/\//.test(githubUrl)) items.push({ label: 'Docs', href: `${githubUrl}#readme`, external: true })
    if (apiUrl && /^https?:\/\//.test(apiUrl)) items.push({ label: 'API Reference', href: apiUrl, external: true })
    if (githubUrl && /^https?:\/\//.test(githubUrl)) items.push({ label: 'GitHub', href: githubUrl, external: true })
    return items
  })
</script>

<template>
  <nav class="devbar" aria-label="Developer shortcuts" v-if="links.length">
    <div class="container">
      <span class="label">Developer</span>
      <ul class="links">
        <li v-for="(l, i) in links" :key="i">
          <a :href="l.href" target="_blank" rel="noopener noreferrer">
            {{ l.label }}
          </a>
        </li>
      </ul>
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
  @media (prefers-color-scheme: dark) {
    .devbar {
      background: color-mix(in srgb, var(--surface) 85%, transparent);
    }
  }
</style>
