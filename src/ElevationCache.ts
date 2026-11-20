import { LngLat } from 'maplibre-gl';
import type { Map } from 'maplibre-gl';

/** Terrain elevation (m) at a point, without the terrain's exaggeration (see createElevationSampler). */
export type ElevationSampler = (lng: number, lat: number) => number;

/** The bits of MapLibre's (internal) `Terrain` the sampler needs. */
interface TerrainInternals {
  exaggeration: number;
  tileManager: { minzoom: number; maxzoom: number };
  /** MapLibre >= 6.6: the rendered DEM tiles, keyed `wrap/z/x/y`, highest zoom first. */
  getCoverageIndex?: () => {
    zooms: number[];
    samplerPerTile: globalThis.Map<string, ((x: number, y: number, extent: number) => number) | null>;
  } | null;
  getElevationForLngLatZoom?: (lngLat: LngLat, zoom: number) => number;
}

const EXTENT = 8192;
const MAX_TILE_COORD = EXTENT * (1 - 1e-12);

/**
 * Elevation lookup for many points under ONE camera state — the same result as
 * `map.queryTerrainElevation`, without its per-call cost.
 *
 * MapLibre samples the rendered DEM tile when a point lies on one; otherwise
 * (DEM not loaded yet, point off the rendered tiles, or always before 6.6) it
 * runs `coveringTiles()` over the whole view on EVERY call. At high pitch that
 * is ~0.5 ms — seconds for a few thousand trees. Here that fallback zoom is
 * resolved once per sampler, so create one per pass and drop it afterwards.
 *
 * The heights come WITHOUT the terrain's exaggeration, as the DEM has them: the
 * layers' vertex shaders multiply them by the exaggeration of the frame they
 * draw, so a terrain whose exaggeration changes (an animation that sets the
 * terrain once per frame) needs no sampling again.
 *
 * Mirrors MapLibre internals (`Terrain.getElevationForLngLat`); falls back to
 * `queryTerrainElevation` when they are missing.
 */
export function createElevationSampler(map: Map): ElevationSampler {
  const sample = createMercatorElevationSampler(map);
  return (lng, lat) =>
    sample(
      (180 + lng) / 360,
      (180 - (180 / Math.PI) * Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360))) / 360,
    );
}

/**
 * {@link createElevationSampler} taking MapLibre mercator units (0..1) instead
 * of lng/lat — for per-vertex lookups, where the geometry already is in them.
 */
export function createMercatorElevationSampler(map: Map): (mx: number, my: number) => number {
  const terrain = map.terrain as unknown as TerrainInternals | null | undefined;
  if (!terrain) return () => 0;
  const toLngLat = (mx: number, my: number) =>
    new LngLat(mx * 360 - 180, (360 / Math.PI) * Math.atan(Math.exp(((180 - my * 360) * Math.PI) / 180)) - 90);
  // MapLibre's own lookups include the exaggeration: taken out again. At an
  // exaggeration of 0 they know nothing of the ground and give 0.
  const unexaggerate = terrain.exaggeration > 0 ? 1 / terrain.exaggeration : 0;
  const exaggerated = terrain.getElevationForLngLatZoom?.bind(terrain);
  const byZoom = exaggerated && ((lngLat: LngLat, zoom: number) => exaggerated(lngLat, zoom) * unexaggerate);
  if (!byZoom) return (mx, my) => (map.queryTerrainElevation(toLngLat(mx, my)) ?? 0) * unexaggerate;

  const index = terrain.getCoverageIndex?.() ?? null;
  let fallbackZoom: number | undefined;
  // The index lookup per (wrap, zoom, tile), memoised under a numeric key: one
  // sampler serves thousands of points in few tiles, and building the string
  // key for every point and zoom was the bulk of a bake. `false` = not covered.
  type TileSample = ((x: number, y: number, extent: number) => number) | null | false;
  const memo = new globalThis.Map<number, TileSample>();
  const direct = (wrap: number, z: number, tx: number, ty: number): TileSample => {
    const key = `${wrap}/${z}/${tx}/${ty}`;
    return index!.samplerPerTile.has(key) ? (index!.samplerPerTile.get(key) ?? null) : false;
  };
  const lookup = (wrap: number, z: number, tx: number, ty: number): TileSample => {
    // Exact in a double for wraps -2..5 and zooms below 22; anything else unmemoised.
    if (wrap < -2 || wrap > 5 || z > 21) return direct(wrap, z, tx, ty);
    const k = ((wrap + 2) * 32 + z) * 2 ** 44 + tx * 2 ** 22 + ty;
    let s = memo.get(k);
    if (s === undefined) memo.set(k, (s = direct(wrap, z, tx, ty)));
    return s;
  };
  // The points of a pass lie close together (a tile's ground columns, the trees
  // of a view): per zoom level, the tile of the point before is tried first —
  // the lookups above were a third of a pass over a tile's columns.
  const zooms = index?.zooms ?? [];
  const lastWrap = new Int32Array(zooms.length).fill(-99);
  const lastTx = new Int32Array(zooms.length);
  const lastTy = new Int32Array(zooms.length);
  const lastSample: TileSample[] = new Array(zooms.length).fill(false);

  return (mx, my) => {
    if (index) {
      if (my >= 0 && my < 1) {
        const wrap = Math.floor(mx);
        for (let i = 0; i < zooms.length; i++) {
          const z = zooms[i];
          const scale = 1 << z;
          const sx = (mx - wrap) * scale;
          const sy = my * scale;
          const tx = Math.floor(sx);
          const ty = Math.floor(sy);
          let sample: TileSample;
          if (tx === lastTx[i] && ty === lastTy[i] && wrap === lastWrap[i]) {
            sample = lastSample[i];
          } else {
            sample = lookup(wrap, z, tx, ty);
            lastWrap[i] = wrap;
            lastTx[i] = tx;
            lastTy[i] = ty;
            lastSample[i] = sample;
          }
          if (sample === false) continue;
          if (sample) {
            return sample(
              Math.min((sx - tx) * EXTENT, MAX_TILE_COORD),
              Math.min((sy - ty) * EXTENT, MAX_TILE_COORD),
              EXTENT,
            );
          }
          break; // covered, but its DEM isn't loaded → fallback
        }
      }
    }
    if (fallbackZoom === undefined) {
      fallbackZoom = 0;
      for (const id of map.coveringTiles({
        tileSize: 512,
        minzoom: terrain.tileManager.minzoom,
        maxzoom: terrain.tileManager.maxzoom,
        terrain: map.terrain,
      } as Parameters<Map['coveringTiles']>[0])) {
        if (id.canonical.z > fallbackZoom) fallbackZoom = Math.min(id.canonical.z, terrain.tileManager.maxzoom);
      }
    }
    return byZoom(toLngLat(mx, my), fallbackZoom);
  };
}

export interface GeoBounds {
  west: number;
  east: number;
  north: number;
  south: number;
}

/** Geographic bounds of a standard web-mercator (slippy) tile. */
export function tileBounds(x: number, y: number, z: number): GeoBounds {
  const n = 2 ** z;
  const lngOf = (xx: number) => (xx / n) * 360 - 180;
  const latOf = (yy: number) =>
    (Math.atan(Math.sinh(Math.PI * (1 - (2 * yy) / n))) * 180) / Math.PI;
  return {
    west: lngOf(x),
    east: lngOf(x + 1),
    north: latOf(y), // y grows southward → y is the northern edge
    south: latOf(y + 1),
  };
}

/**
 * The map's terrain source, `''` without terrain: what the layers' heights
 * depend on besides the DEM data. The exaggeration is not part of it — the
 * samplers above leave it out and the vertex shaders apply it per frame.
 */
export function terrainKeyOf(map: Map): string {
  return map.getTerrain()?.source ?? '';
}

/** The exaggeration the map's terrain draws with in this frame (1 without terrain). */
export function exaggerationOf(map: Map): number {
  return (map.terrain as unknown as TerrainInternals | null | undefined)?.exaggeration ?? 1;
}
