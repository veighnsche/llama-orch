import type { NextConfig } from "next";
import nextra from "nextra";

const nextConfig: NextConfig = {
  images: {
    unoptimized: true, // Cloudflare Workers compatibility
  },
};

const withNextra = nextra({
  // Nextra configuration options
  defaultShowCopyCode: true,
  search: {
    codeblocks: false,
  },
});

export default withNextra(nextConfig);

// added by create cloudflare to enable calling `getCloudflareContext()` in `next dev`
import { initOpenNextCloudflareForDev } from '@opennextjs/cloudflare';
initOpenNextCloudflareForDev();
