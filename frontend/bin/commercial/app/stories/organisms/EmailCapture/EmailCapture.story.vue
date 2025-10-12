<!-- TEAM-FE-004: Converted from .story.ts to .story.vue format -->
<!-- TEAM-FE-CONSOLIDATE: Added interactive variants -->
<script setup lang="ts">
import EmailCapture from './EmailCapture.vue'
</script>

<template>
  <Story title="organisms/EmailCapture">
    <Variant title="Default">
      <div class="bg-background">
        <EmailCapture />
      </div>
    </Variant>

    <Variant title="Custom Content">
      <div class="bg-background">
        <EmailCapture
          title="Get Early Access"
          description="Be among the first to experience rbee. Sign up now for exclusive beta access and special launch pricing."
          status-badge="Beta Launch Coming Soon"
          privacy-text="Your email is safe with us. No spam, ever."
          github-text="Follow Development on GitHub"
        />
      </div>
    </Variant>

    <Variant title="Minimal">
      <div class="bg-background">
        <EmailCapture
          title="Stay Updated"
          description="Get notified when we launch."
          status-badge="Coming Soon"
          privacy-text="We respect your privacy."
        />
      </div>
    </Variant>
  </Story>
</template>
