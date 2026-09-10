import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

// One config, three jobs, selected by BUILD_TARGET:
//   (unset)      → dev server + demo app build (index.html)
//   'npm'        → ESM library build for the npm package (peer/deps external)
//   'standalone' → self-contained ESM for <script type=module> + import map
const target = process.env.BUILD_TARGET;

const entry = fileURLToPath(new URL('./src/index.ts', import.meta.url));

// Kept out of the ESM bundle so the consumer resolves (and dedupes) one copy.
// `three` + `maplibre-gl` are peerDependencies; the rest are regular deps.
const externalIds = ['three', 'maplibre-gl', 'earcut', 'pbf', '@maplibre/maplibre-gl-style-spec'];
const isExternal = (id: string) => externalIds.some((p) => id === p || id.startsWith(p + '/'));

export default defineConfig(() => {
  // Worker is authored as an ES module and inlined as a blob (?worker&inline),
  // so no separate worker asset is emitted in any build.
  const common = { worker: { format: 'es' as const } };

  if (target === 'npm') {
    return {
      ...common,
      build: {
        emptyOutDir: true,
        lib: { entry, formats: ['es' as const], fileName: () => 'maplibre-3d-layers.js' },
        rollupOptions: { external: isExternal },
      },
    };
  }

  if (target === 'standalone') {
    return {
      ...common,
      build: {
        emptyOutDir: false, // runs after the npm build — keep its output
        lib: {
          entry,
          formats: ['es' as const],
          fileName: () => 'maplibre-3d-layers.standalone.mjs',
        },
        // Self-contained ESM for <script type="module"> + an import map: three,
        // earcut, pbf and the style-spec are bundled in (the worker is inlined
        // too), so ONLY maplibre-gl stays external. maplibre-gl v6 is ESM-only —
        // there is no global build — so the page supplies it via a one-entry
        // import map, not a <script> global.
        rollupOptions: { external: ['maplibre-gl'] },
      },
    };
  }

  // Dev server + demo app build.
  return {
    ...common,
    // Vite's esbuild dep-optimizer breaks maplibre-gl's internal web worker
    // (maplibre-gl-worker.mjs). Excluding it keeps the worker loadable.
    optimizeDeps: { exclude: ['maplibre-gl'] },
    server: { open: false },
  };
});
