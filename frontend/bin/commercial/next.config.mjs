/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: ['@rbee/shared-components'],
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
};

export default nextConfig
