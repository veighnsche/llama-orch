<script setup lang="ts">
import { useI18n } from 'vue-i18n'
const { t } = useI18n()
import Button from 'orchyra-storybook/stories/button.vue'
import Badge from '@ui/badge'
</script>

<template>
    <section class="why" aria-labelledby="why-title">
        <div class="why-head">
            <h3 id="why-title">
                {{ t('home.why.title') }}
            </h3>
            <p class="why-intro">
                {{ t('home.why.intro') }}
            </p>
        </div>

        <div class="why-grid">
            <!-- Column: problems -->
            <div class="panel">
                <h4 class="eyebrow">
                    {{ t('home.why.problemsTitle') }}
                </h4>
                <ul class="list">
                    <li>
                        <span class="dot" aria-hidden="true" />
                        {{ t('home.why.b1') }}
                    </li>
                    <li>
                        <span class="dot" aria-hidden="true" />
                        {{ t('home.why.b2') }}
                    </li>
                    <li>
                        <span class="dot" aria-hidden="true" />
                        {{ t('home.why.b3') }}
                    </li>
                    <li v-if="$te('home.why.b4')">
                        <span class="dot" aria-hidden="true" />
                        {{ t('home.why.b4') }}
                    </li>
                </ul>
            </div>

            <!-- Column: outcomes -->
            <div class="panel">
                <h4 class="eyebrow">
                    {{ t('home.why.outcomesTitle') }}
                </h4>
                <ul class="list">
                    <li>
                        <span class="check" aria-hidden="true">✓</span>
                        {{ t('home.why.o1') }}
                    </li>
                    <li>
                        <span class="check" aria-hidden="true">✓</span>
                        {{ t('home.why.o2') }}
                    </li>
                    <li>
                        <span class="check" aria-hidden="true">✓</span>
                        {{ t('home.why.o3') }}
                    </li>
                    <li v-if="$te('home.why.o4')">
                        <span class="check" aria-hidden="true">✓</span>
                        {{ t('home.why.o4') }}
                    </li>
                </ul>

                <div class="badges" aria-label="proof badges">
                    <Badge variant="green">
                        {{ t('home.badge.audit') }}
                    </Badge>
                    <Badge variant="cyan">
                        {{ t('home.badge.gdpr') }}
                    </Badge>
                    <Badge variant="slate">
                        {{ t('home.badge.oss') }}
                    </Badge>
                </div>

                <div class="cta-row">
                    <Button as="router-link" to="/proof" variant="primary">
                        {{ t('home.why.ctaProof') }}
                    </Button>
                    <Button as="router-link" to="/service-menu" variant="ghost">
                        {{ t('home.why.ctaMenu') }}
                    </Button>
                </div>
            </div>
        </div>
    </section>
</template>

<style scoped>
.why {
    margin: 1.75rem 0;
}
.why-head {
    margin-bottom: 0.75rem;
}
#why-title {
    margin: 0 0 0.25rem 0;
}
.why-intro {
    color: var(--muted);
    margin: 0;
}

.why-grid {
    display: grid;
    gap: 1rem;
}
.panel {
    border: 1px solid var(--surface-muted);
    border-radius: var(--radius-lg);
    padding: 0.9rem;
    background: var(--surface-alt);
}
.eyebrow {
    font-size: 0.9rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--acc-cyan);
    margin: 0 0 0.4rem 0;
}
.list {
    list-style: none;
    padding: 0;
    margin: 0;
    display: grid;
    gap: 0.35rem;
}
.list li {
    display: flex;
    align-items: start;
    gap: 0.5rem;
    color: var(--text);
}
.dot {
    width: 0.5rem;
    height: 0.5rem;
    border-radius: var(--radius-pill);
    background: var(--muted);
    margin-top: 0.4rem;
}
.check {
    color: var(--ok-green);
    font-weight: 800;
    margin-top: 0.05rem;
}

.badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.6rem;
}

.cta-row {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.75rem;
}

@media (min-width: 920px) {
    .why-grid {
        grid-template-columns: 1fr 1fr;
    }
}
@media (prefers-color-scheme: dark) {
    .why-intro {
        color: var(--muted);
    }
    .panel {
        background: var(--surface);
        border-color: var(--surface-muted);
    }
    .list li {
        color: var(--text);
    }
    .dot {
        background: var(--muted);
    }
    /* Badges now use ui/Badge.vue which handles dark mode */
    /* Button component has dark styles built-in; no overrides needed here */
}
</style>
