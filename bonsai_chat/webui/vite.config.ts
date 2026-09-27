import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

// `base: './'` is load-bearing: the built index.html is served both from `/` and from
// `/ui/`, so every asset reference must resolve relative to the document, never to a
// fixed origin root.
export default defineConfig({
  base: './',
  plugins: [react()],
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false,
    target: 'es2020',
  },
  test: {
    environment: 'jsdom',
    globals: true,
    include: ['tests/**/*.test.{ts,tsx}'],
    setupFiles: ['tests/setup.ts'],
  },
});
