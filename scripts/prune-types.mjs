// Ship type declarations for the public API only. `tsc` emits one .d.ts per
// module; everything the published entry point does not reference is dropped
// again here, so the package stays fully typed without carrying the internals.
import { readdir, readFile, unlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const DIST = new URL('../dist/', import.meta.url).pathname;
const PUBLIC = ['index', 'Fill3DLayer', 'GltfLayer', 'expression', 'paint'];
const keep = new Set(PUBLIC.map((n) => `${n}.d.ts`));

const files = (await readdir(DIST)).filter((f) => f.endsWith('.d.ts'));
const removed = [];
for (const file of files) {
  if (keep.has(file)) continue;
  await unlink(join(DIST, file));
  removed.push(file);
}

// Doc comments on private members explain the implementation to us, not the
// API to consumers — tsc emits them anyway, so drop them here.
let stripped = 0;
for (const file of keep) {
  const source = await readFile(join(DIST, file), 'utf8');
  const cleaned = source.replace(/[ \t]*\/\*\*[\s\S]*?\*\/\n(?=[ \t]*private\s)/g, () => {
    stripped++;
    return '';
  });
  if (cleaned !== source) await writeFile(join(DIST, file), cleaned);
}

// Guard: a kept declaration must never import one we just removed, or consumers
// get a broken type graph.
for (const file of keep) {
  const source = await readFile(join(DIST, file), 'utf8');
  for (const [, spec] of source.matchAll(/from '\.\/([\w-]+)'/g)) {
    if (!keep.has(`${spec}.d.ts`)) {
      console.error(`prune-types: ${file} imports './${spec}', which is not published`);
      process.exit(1);
    }
  }
}

console.log(`prune-types: kept ${keep.size}, removed ${removed.length}, stripped ${stripped} private doc blocks`);
