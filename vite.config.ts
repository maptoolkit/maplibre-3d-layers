import { defineConfig } from 'vite';

// Worker is authored as an ES module (`{ type: 'module' }`).
export default defineConfig({
  worker: { format: 'es' },
  // Vite's esbuild dep-optimizer breaks maplibre-gl's internal web worker
  // (maplibre-gl-worker.mjs). Excluding it keeps the worker loadable.
  optimizeDeps: { exclude: ['maplibre-gl'] },
  server: { open: false },
});
