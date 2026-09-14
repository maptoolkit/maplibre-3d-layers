import { PbfReader as Pbf } from 'pbf';
import earcut from 'earcut';
import { featureFilter } from '@maplibre/maplibre-gl-style-spec';
import type { FilterSpecification, Feature, GlobalProperties } from '@maplibre/maplibre-gl-style-spec';
import { VectorTile } from './vectorTile3d';

/** Request: the fetched tile plus which layer to read and an optional filter.
 *  Geometry is stored tile-locally, so the scene origin is NOT needed here. */
export interface TileRequest {
  key: string;
  x: number;
  y: number;
  z: number;
  buffer: ArrayBuffer;
  /** Vector-tile layer name to read (the `source-layer`). */
  sourceLayer: string;
  /** Optional MapLibre filter expression; features that fail are dropped. */
  filter?: FilterSpecification;
}

/** One feature's vertex span, so the main thread can bake terrain + colour per feature. */
export interface FeatureSpan {
  lng: number;
  lat: number;
  /** First vertex index (into `position`, in vertices) and vertex count. */
  start: number;
  count: number;
  /** Feature properties, for evaluating paint expressions on the main thread. */
  properties: Record<string, number | string | boolean>;
}

export interface TileResult {
  key: string;
  /**
   * xyz per vertex in TRUE METRES relative to the tile's NW corner:
   * x=east, y=height-above-base, z=north. Scene-origin independent.
   */
  position: Float32Array;
  /** Per-vertex face normal in the same (east, up, north) frame. */
  normal: Float32Array;
  index: Uint32Array;
  features: FeatureSpan[];
  /** meterInMercatorCoordinateUnits at the tile's latitude (for placement). */
  tileScale: number;
}

interface P {
  x: number; // east (metres)
  y: number; // up (base-relative metres)
  z: number; // north (metres)
}

// MapLibre's mercator metric (earthRadius 6371008.8 m).
const EARTH_CIRCUMFERENCE = 2 * Math.PI * 6371008.8;

/** meterInMercatorCoordinateUnits() equivalent for a given latitude. */
function mercatorScaleAtLat(latDeg: number): number {
  return 1 / (EARTH_CIRCUMFERENCE * Math.cos((latDeg * Math.PI) / 180));
}

/** Parse + triangulate one tile's 3D-feature layer into tile-local metre geometry. */
export function buildTile(
  tx: number,
  ty: number,
  tz: number,
  buffer: ArrayBuffer,
  sourceLayer: string,
  filter?: FilterSpecification,
): Omit<TileResult, 'key'> {
  const pow = 2 ** tz;
  // Metric at the tile's centre latitude (geometry is stored relative to the corner).
  const tileLat = mercToLngLat((tx + 0.5) / pow, (ty + 0.5) / pow)[1];
  const tileScale = mercatorScaleAtLat(tileLat);

  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];
  const features: FeatureSpan[] = [];

  const tile = new VectorTile(new Pbf(new Uint8Array(buffer)));
  const layer = tile.layers[sourceLayer];

  const ff = filter ? featureFilter(filter, 'filter') : null;
  const globals = { zoom: tz } as GlobalProperties;

  if (layer) {
    for (let i = 0; i < layer.length; i++) {
      const feature = layer.feature(i);
      if (feature.type !== 0) continue; // only the custom 3D geometry
      if (ff && !ff.filter(globals, feature as unknown as Feature)) continue;

      const polygons = feature.loadGeometry();
      if (polygons.length < 2) continue; // need bbox + at least one real polygon

      // First polygon is a bounding box: [min, max] corners → base z + centroid.
      const bboxRing = polygons.shift()![0];
      const baseZ = bboxRing[0].z;
      const cx = (bboxRing[0].x + bboxRing[1].x) / 2;
      const cy = (bboxRing[0].y + bboxRing[1].y) / 2;

      // De-dupe across tile boundaries. The source emits each building WHOLE into
      // every tile its geometry overlaps (geometry overhangs a tile edge by
      // hundreds of extent units), and feature ids are renumbered per tile, so at
      // shared edges/corners the same building sits in 2–4 tiles. Drawing every
      // copy double-draws coplanar walls → z-fighting/flicker. Render a building
      // only in the tile whose bounds contain its (tile-stable) bbox centroid.
      // Half-open [0, extent) → an exact partition (verified: 0 buildings owned by
      // >1 tile across a 2×2 z15 block); also drops ~¼ of features as duplicates.
      const ext = feature.extent;
      if (cx < 0 || cx >= ext || cy < 0 || cy >= ext) continue;

      const [lng, lat] = mercToLngLat(
        (tx + cx / feature.extent) / pow,
        (ty + cy / feature.extent) / pow,
      );

      const featureStart = positions.length / 3;

      for (const polygon of polygons) {
        const polyVertexBase = positions.length / 3;
        const flat2D: number[] = [];
        const holes: number[] = [];

        for (let r = 0; r < polygon.length; r++) {
          if (r) holes.push(flat2D.length / 2);
          const ring = polygon[r];
          const pts = ring.map((v) => toTileLocal(v, baseZ, feature.extent, pow, tileScale));
          const n = newellNormal(pts);
          for (const p of pts) {
            positions.push(p.x, p.y, p.z);
            normals.push(n[0], n[1], n[2]);
            const q = to2D(p, n);
            flat2D.push(q[0], q[1]);
          }
        }

        const tri = flat2D.length === 6 ? [0, 1, 2] : earcut(flat2D, holes, 2);
        for (let t = 0; t < tri.length; t++) indices.push(tri[t] + polyVertexBase);
      }

      features.push({
        lng,
        lat,
        start: featureStart,
        count: positions.length / 3 - featureStart,
        properties: feature.properties,
      });
    }
  }

  return {
    position: new Float32Array(positions),
    normal: new Float32Array(normals),
    index: new Uint32Array(indices),
    features,
    tileScale,
  };
}

/** Tile-local vertex → true metres (east, up-above-base, north) from the tile NW corner. */
function toTileLocal(
  v: { x: number; y: number; z: number },
  baseZ: number,
  extent: number,
  pow: number,
  tileScale: number,
): P {
  const mx = (v.x / extent) / pow; // mercator offset east of the tile corner
  const my = (v.y / extent) / pow; // mercator offset south of the tile corner
  return {
    x: mx / tileScale, // east metres
    y: (v.z - baseZ) / 10, // decimetres → metres, base at 0 (terrain added later)
    z: -my / tileScale, // north metres (mercator y grows southward)
  };
}

/** Newell's method — face normal robust to non-planar rings. */
function newellNormal(pts: P[]): [number, number, number] {
  const n: [number, number, number] = [0, 0, 0];
  const len = pts.length - 1;
  for (let i = 0; i <= len; i++) {
    const p0 = pts[i];
    const p1 = pts[i === len ? 0 : i + 1];
    n[0] += (p0.y - p1.y) * (p0.z + p1.z);
    n[1] += (p0.z - p1.z) * (p0.x + p1.x);
    n[2] += (p0.x - p1.x) * (p0.y + p1.y);
  }
  const l = Math.hypot(n[0], n[1], n[2]) || 1;
  n[0] /= l;
  n[1] /= l;
  n[2] /= l;
  return n;
}

/** Project a 3D point onto the plane ⟂ normal, giving 2D coords for earcut. */
function to2D(p: P, n: [number, number, number]): [number, number] {
  const px = [p.x, p.y, p.z];
  const x = [1.1, 1.1, 1.1];
  const d = x[0] * n[0] + x[1] * n[1] + x[2] * n[2];
  x[0] -= n[0] * d;
  x[1] -= n[1] * d;
  x[2] -= n[2] * d;
  const xl = Math.hypot(x[0], x[1], x[2]) || 1;
  x[0] /= xl;
  x[1] /= xl;
  x[2] /= xl;
  const y = [
    n[1] * x[2] - n[2] * x[1],
    n[2] * x[0] - n[0] * x[2],
    n[0] * x[1] - n[1] * x[0],
  ];
  return [
    px[0] * x[0] + px[1] * x[1] + px[2] * x[2],
    px[0] * y[0] + px[1] * y[1] + px[2] * y[2],
  ];
}

/** Inverse web-mercator (normalized 0..1) → [lng, lat] degrees. */
function mercToLngLat(mx: number, my: number): [number, number] {
  const lng = mx * 360 - 180;
  const t = 180 - my * 360;
  const lat = (360 / Math.PI) * Math.atan(Math.exp((t * Math.PI) / 180)) - 90;
  return [lng, lat];
}
