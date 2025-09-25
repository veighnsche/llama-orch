<script setup lang="ts">
import { useMeta } from '@/composables/useMeta'
import { useI18n } from 'vue-i18n'
import { useRoute } from 'vue-router'

const { t, tm, locale } = useI18n()
const route = useRoute()

useMeta({
    title: () => t('seoTitle.faqs'),
    description: () => t('seoDesc.faqs'),
    keywords: () => tm('seo.faqs') as string[],
    canonical: () => `${window.location.origin}${route.fullPath}`,
    alternates: () => {
        const href = `${window.location.origin}${route.fullPath}`
        return [
            { hrefLang: 'nl', href },
            { hrefLang: 'en', href },
            { hrefLang: 'x-default', href },
        ]
    },
    watchSources: [() => locale.value, () => route.fullPath],
})
</script>

<template>
    <main class="page">
        <h1>{{ $t('faqs.h1') }}</h1>

        <section>
            <h2>{{ $t('faqs.q1') }}</h2>
            <p>{{ $t('faqs.a1') }}</p>
        </section>

        <section>
            <h2>{{ $t('faqs.q2') }}</h2>
            <p>{{ $t('faqs.a2') }}</p>
        </section>

        <section>
            <h2>{{ $t('faqs.q3') }}</h2>
            <p>{{ $t('faqs.a3') }}</p>
        </section>

        <section>
            <h2>{{ $t('faqs.q4') }}</h2>
            <p>{{ $t('faqs.a4') }}</p>
        </section>

        <section>
            <h2>{{ $t('faqs.q5') }}</h2>
            <p>{{ $t('faqs.a5') }}</p>
        </section>
    </main>
</template>

<style scoped>
.page {
    display: grid;
    gap: 1rem;
}
</style>
