import type { ExtrusionRequest } from './mvt3dGeometry';

/** Per vector-tile layer of a source: the zoom levels its tiles carry it at (TileJSON `vector_layers`). */
export type LayerZooms = ReadonlyMap<string, readonly [number, number]>;

/**
 * The tile zoom levels a filter can pass at: what an `["all", …]` of
 * comparisons of `["zoom"]` with a number allows (a tile's zoom is whole).
 * Anything else in the filter does not narrow the range. Empty: lo > hi.
 */
export function filterZooms(filter: unknown): [number, number] {
  let lo = -Infinity;
  let hi = Infinity;
  const isZoom = (v: unknown) => Array.isArray(v) && v.length === 1 && v[0] === 'zoom';
  const mirrored: Record<string, string> = { '<': '>', '<=': '>=', '>': '<', '>=': '<=', '==': '==' };
  const visit = (f: unknown) => {
    if (!Array.isArray(f)) return;
    if (f[0] === 'all') {
      for (let i = 1; i < f.length; i++) visit(f[i]);
      return;
    }
    if (f.length !== 3 || typeof f[0] !== 'string') return;
    let op = f[0];
    let n: number;
    if (isZoom(f[1]) && typeof f[2] === 'number') n = f[2];
    else if (isZoom(f[2]) && typeof f[1] === 'number') {
      n = f[1];
      op = mirrored[op] ?? '';
    } else return;
    if (op === '>=') lo = Math.max(lo, Math.ceil(n));
    else if (op === '>') lo = Math.max(lo, Math.floor(n) + 1);
    else if (op === '<=') hi = Math.min(hi, Math.floor(n));
    else if (op === '<') hi = Math.min(hi, Math.ceil(n) - 1);
    else if (op === '==') {
      if (Number.isInteger(n)) {
        lo = Math.max(lo, n);
        hi = Math.min(hi, n);
      } else lo = Infinity;
    }
  };
  visit(filter);
  return [lo, hi];
}

/**
 * The tile zoom levels at which a Fill3DLayer can have anything to draw: where
 * its source layer exists and its filter can pass, likewise for its extrusion
 * layer, together. Tiles of other zoom levels need not be loaded (in a pitched
 * view MapLibre covers the distance with tiles of lower zoom levels: with
 * `buildings3d` at 15 only, about half of the covering tiles). Undefined: nothing at all.
 */
export function dataZooms(
  layers: LayerZooms | undefined,
  sourceLayer: string,
  filter: unknown,
  extrusion: ExtrusionRequest | undefined,
  minzoom: number,
  maxzoom: number,
): [number, number] | undefined {
  const range = (layer: string, f: unknown): [number, number] => {
    const [lmin, lmax] = layers?.get(layer) ?? [minzoom, maxzoom];
    const [fmin, fmax] = filterZooms(f);
    return [Math.max(minzoom, lmin, fmin), Math.min(maxzoom, lmax, fmax)];
  };
  const ranges = [range(sourceLayer, filter)];
  if (extrusion) ranges.push(range(extrusion.sourceLayer, extrusion.filter));
  const live = ranges.filter(([lo, hi]) => lo <= hi);
  if (!live.length) return undefined;
  return [Math.min(...live.map((r) => r[0])), Math.max(...live.map((r) => r[1]))];
}
