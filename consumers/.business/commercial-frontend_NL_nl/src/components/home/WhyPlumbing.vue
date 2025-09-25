<script setup lang="ts">
import { useI18n } from 'vue-i18n'
const { t } = useI18n()
import Button from '@/components/ui/Button.vue'
import Badge from '@/components/ui/Badge.vue'
</script>

<template>
  <section class="why" aria-labelledby="why-title">
    <div class="why-head">
      <h3 id="why-title">{{ t('home.why.title') }}</h3>
      <p class="why-intro">
        {{ t('home.why.intro') }}
      </p>
    </div>

    <div class="why-grid">
      <!-- Column: problems -->
      <div class="panel">
        <h4 class="eyebrow">{{ t('home.why.problemsTitle') }}</h4>
        <ul class="list">
          <li>
            <span class="dot" aria-hidden="true"></span>
            {{ t('home.why.b1') }}
          </li>
          <li>
            <span class="dot" aria-hidden="true"></span>
            {{ t('home.why.b2') }}
          </li>
          <li>
            <span class="dot" aria-hidden="true"></span>
            {{ t('home.why.b3') }}
          </li>
          <li v-if="$te('home.why.b4')">
            <span class="dot" aria-hidden="true"></span>
            {{ t('home.why.b4') }}
          </li>
        </ul>
      </div>

      <!-- Column: outcomes -->
      <div class="panel">
        <h4 class="eyebrow">{{ t('home.why.outcomesTitle') }}</h4>
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
          <Badge variant="green">{{ t('home.badge.audit') }}</Badge>
          <Badge variant="cyan">{{ t('home.badge.gdpr') }}</Badge>
          <Badge variant="slate">{{ t('home.badge.oss') }}</Badge>
        </div>

        <div class="cta-row">
          <Button as="router-link" to="/proof" variant="primary">{{ t('home.why.ctaProof') }}</Button>
          <Button as="router-link" to="/service-menu" variant="ghost">{{ t('home.why.ctaMenu') }}</Button>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.why { margin: 1.75rem 0; }
.why-head { margin-bottom: .75rem; }
#why-title { margin: 0 0 .25rem 0; }
.why-intro { color: #334155; margin: 0; }

.why-grid {
  display: grid;
  gap: 1rem;
}
.panel {
  border: 1px solid var(--color-border, #cbd5e1);
  border-radius: 10px;
  padding: .9rem;
  background: #ffffff;
}
.eyebrow {
  font-size: .9rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .04em;
  color: #0ea5e9;
  margin: 0 0 .4rem 0;
}
.list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: grid;
  gap: .35rem;
}
.list li { display: flex; align-items: start; gap: .5rem; color: #0f172a; }
.dot {
  width: .5rem; height: .5rem; border-radius: 999px; background: #94a3b8; margin-top: .4rem;
}
.check { color: #059669; font-weight: 800; margin-top: .05rem; }

.badges { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .6rem; }

.cta-row { display: flex; gap: .5rem; margin-top: .75rem; }

@media (min-width: 920px) {
  .why-grid { grid-template-columns: 1fr 1fr; }
}
@media (prefers-color-scheme: dark) {
  .why-intro { color: #cbd5e1; }
  .panel { background: #0b1220; border-color: #0b1b2e; }
  .list li { color: #e5f1ff; }
  .dot { background: #64748b; }
  /* Badges now use ui/Badge.vue which handles dark mode */
  /* Button component has dark styles built-in; no overrides needed here */
}
</style>
