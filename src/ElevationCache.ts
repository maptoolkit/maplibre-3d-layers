import { LngLat } from 'maplibre-gl';
import type { Map, LngLatLike } from 'maplibre-gl';

interface Entry {
  lngLat: LngLat;
  elevation: number;
}

/** Terrain elevation (m) at a point. */
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
  const byZoom = terrain.getElevationForLngLatZoom?.bind(terrain);
  if (!byZoom) return (mx, my) => map.queryTerrainElevation(toLngLat(mx, my)) ?? 0;

  const index = terrain.getCoverageIndex?.() ?? null;
  let fallbackZoom: number | undefined;

  return (mx, my) => {
    if (index) {
      if (my >= 0 && my < 1) {
        const wrap = Math.floor(mx);
        for (const z of index.zooms) {
          const scale = 1 << z;
          const sx = (mx - wrap) * scale;
          const sy = my * scale;
          const tx = Math.floor(sx);
          const ty = Math.floor(sy);
          const key = `${wrap}/${z}/${tx}/${ty}`;
          if (!index.samplerPerTile.has(key)) continue;
          const sample = index.samplerPerTile.get(key);
          if (sample) {
            return sample(
              Math.min((sx - tx) * EXTENT, MAX_TILE_COORD),
              Math.min((sy - ty) * EXTENT, MAX_TILE_COORD),
              EXTENT,
            ) * terrain.exaggeration;
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

/**
 * Caches `map.queryTerrainElevation` results per sample point so the render loop
 * doesn't re-query every frame.
 *
 * Terrain DEM tiles stream in asynchronously (and can be replaced when zooming),
 * so a cached elevation goes stale the moment a fresh tile covering that point
 * arrives. We listen for the source's per-tile `data` events and invalidate only
 * the cached samples that fall inside the newly (re)loaded tile's geographic
 * bounds — everything else stays cached.
 */
export class ElevationCache {
  private readonly map: Map;
  private readonly sourceId: string;
  private readonly entries = new globalThis.Map<string, Entry>();
  private readonly onData: (e: unknown) => void;

  constructor(map: Map, sourceId: string) {
    this.map = map;
    this.sourceId = sourceId;

    this.onData = (e) => {
      // Only per-tile source data for our terrain source carries a tileID.
      const ev = e as {
        sourceId?: string;
        tile?: { tileID?: { canonical?: { x: number; y: number; z: number } } };
      };
      if (ev.sourceId !== this.sourceId) return;
      const c = ev.tile?.tileID?.canonical;
      if (!c) return;
      this.invalidateTile(c.x, c.y, c.z);
    };
    map.on('data', this.onData);
  }

  /** Detach the terrain-data listener and clear the cache. */
  dispose() {
    this.map.off('data', this.onData);
    this.entries.clear();
  }

  /** Ground elevation (m) at a point; queries + caches on a miss. */
  get(point: LngLatLike): number {
    const lngLat = LngLat.convert(point);
    const key = ElevationCache.key(lngLat.lng, lngLat.lat);
    const hit = this.entries.get(key);
    if (hit) return hit.elevation;

    const elevation = this.map.queryTerrainElevation(lngLat) ?? 0;
    this.entries.set(key, { lngLat, elevation });
    return elevation;
  }

  /**
   * Like {@link get}, for many points in one synchronous pass: misses share one
   * {@link createElevationSampler}. Drop the returned function after the pass.
   */
  batch(): (lngLat: LngLat) => number {
    let sample: ElevationSampler | undefined;
    return (lngLat) => {
      const key = ElevationCache.key(lngLat.lng, lngLat.lat);
      const hit = this.entries.get(key);
      if (hit) return hit.elevation;

      sample ??= createElevationSampler(this.map);
      const elevation = sample(lngLat.lng, lngLat.lat);
      this.entries.set(key, { lngLat, elevation });
      return elevation;
    };
  }

  /** Drop cached samples inside the given tile's bounds so they re-query. */
  private invalidateTile(x: number, y: number, z: number) {
    const b = tileBounds(x, y, z);
    let changed = false;
    for (const [key, entry] of this.entries) {
      const { lng, lat } = entry.lngLat;
      if (lng >= b.west && lng <= b.east && lat >= b.south && lat <= b.north) {
        this.entries.delete(key);
        changed = true;
      }
    }
    // Fresh elevations available → let the custom layer re-render with them.
    if (changed) this.map.triggerRepaint();
  }

  private static key(lng: number, lat: number): string {
    // ~0.1 m grid — enough to dedupe identical sample points without
    // collapsing distinct feature centroids.
    return `${lng.toFixed(6)},${lat.toFixed(6)}`;
  }
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
