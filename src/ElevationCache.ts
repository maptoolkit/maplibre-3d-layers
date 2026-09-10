import { LngLat } from 'maplibre-gl';
import type { Map, LngLatLike } from 'maplibre-gl';

interface Entry {
  lngLat: LngLat;
  elevation: number;
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

/** Geographic bounds of a standard web-mercator (slippy) tile. */
function tileBounds(x: number, y: number, z: number) {
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
