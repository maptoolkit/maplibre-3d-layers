import { PbfReader as Pbf } from 'pbf';
import earcut from 'earcut';
import { featureFilter } from '@maplibre/maplibre-gl-style-spec';
import type { FilterSpecification, Feature, GlobalProperties } from '@maplibre/maplibre-gl-style-spec';
import { VectorTile } from './vectorTile3d';
import type { Polygon } from './vectorTile3d';

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
  /** Per-vertex face normal in the same (east, up, north) frame, scaled to ±127
   *  (a normalized Int8 attribute — a quarter of the bytes of float32). */
  normal: Int8Array;
  /** Uint16 while the tile has at most 65536 vertices (half the bytes), else Uint32. */
  index: Uint16Array | Uint32Array;
  features: FeatureSpan[];
  /** meterInMercatorCoordinateUnits at the tile's latitude (for placement). */
  tileScale: number;
}

interface P {
  x: number; // east (metres)
  y: number; // up (base-relative metres)
  z: number; // north (metres)
}

/** One 3D feature of the tile, its bounding-box polygon already removed. */
interface SourceFeature {
  properties: Record<string, number | string | boolean>;
  extent: number;
  baseZ: number;
  polygons: Polygon[];
}

/** A connected building: its faces, per source feature, and their extent. */
interface Part {
  members: { feature: SourceFeature; polygons: Polygon[] }[];
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

/**
 * Faces closer than this (metres, between their bounding boxes) belong to the
 * same building. Roof and walls of one house touch; separate buildings bundled
 * into one feature are metres to kilometres apart.
 */
const PART_GAP_M = 2;

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
    // A building arrives as a roof and a wall feature with the same id, each a
    // multipolygon of single faces. The source may also bundle several separate
    // buildings into one feature (features with identical attributes get merged
    // into a multipolygon). Terrain has to be anchored per BUILDING: per face
    // would tear a roof from its walls on a slope, per feature would put a whole
    // bundle on the height of its common centre. So: collect the faces of all
    // features sharing an id, split them into connected parts, and anchor each
    // part at its own centre — roof and walls of one house share that anchor.
    const byId = new globalThis.Map<unknown, SourceFeature[]>();
    for (let i = 0; i < layer.length; i++) {
      const feature = layer.feature(i);
      if (feature.type !== 0) continue; // only the custom 3D geometry
      if (ff && !ff.filter(globals, feature as unknown as Feature)) continue;

      const polygons = feature.loadGeometry();
      if (polygons.length < 2) continue; // need bbox + at least one real polygon

      // First polygon is a bounding box: [min, max] corners → base z.
      const baseZ = polygons.shift()![0][0].z;
      const id = feature.properties.id ?? `\u0000${i}`; // no id: a group of its own
      let group = byId.get(id);
      if (!group) byId.set(id, (group = []));
      group.push({ properties: feature.properties, extent: feature.extent, baseZ, polygons });
    }

    for (const group of byId.values()) {
      const extent = group[0].extent;
      const tolerance = PART_GAP_M * extent * pow * tileScale; // metres → extent units
      for (const part of connectedParts(group, tolerance)) {
        const cx = (part.minX + part.maxX) / 2;
        const cy = (part.minY + part.maxY) / 2;

        // De-dupe across tile boundaries. The source emits each building WHOLE
        // into every tile its geometry overlaps (geometry overhangs a tile edge by
        // hundreds of extent units), and feature ids are renumbered per tile, so at
        // shared edges/corners the same building sits in 2–4 tiles. Drawing every
        // copy double-draws coplanar walls → z-fighting/flicker. Render a building
        // only in the tile whose bounds contain its (tile-stable) centre.
        // Half-open [0, extent) → an exact partition.
        if (cx < 0 || cx >= extent || cy < 0 || cy >= extent) continue;

        const [lng, lat] = mercToLngLat((tx + cx / extent) / pow, (ty + cy / extent) / pow);

        for (const member of part.members) {
          const src = member.feature;
          const featureStart = positions.length / 3;
          for (const polygon of member.polygons) {
            const polyVertexBase = positions.length / 3;
            const flat2D: number[] = [];
            const holes: number[] = [];

            for (let r = 0; r < polygon.length; r++) {
              if (r) holes.push(flat2D.length / 2);
              const ring = polygon[r];
              const pts = ring.map((v) => toTileLocal(v, src.baseZ, src.extent, pow, tileScale));
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
            properties: src.properties,
          });
        }
      }
    }
  }

  const vertexCount = positions.length / 3;
  const normal = new Int8Array(normals.length);
  for (let i = 0; i < normals.length; i++) normal[i] = Math.round(normals[i] * 127);
  return {
    position: new Float32Array(positions),
    normal,
    index: vertexCount <= 65536 ? new Uint16Array(indices) : new Uint32Array(indices),
    features,
    tileScale,
  };
}

/**
 * Split the faces of a building id's features into connected parts: faces
 * whose bounding boxes lie within `tolerance` (extent units) of each other,
 * transitively. A sweep over the boxes sorted by x keeps large city-model
 * buildings (thousands of faces) cheap.
 */
function connectedParts(group: SourceFeature[], tolerance: number): Part[] {
  const faces: { feature: SourceFeature; polygon: Polygon; box: [number, number, number, number] }[] = [];
  for (const feature of group) {
    for (const polygon of feature.polygons) {
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const v of polygon[0]) {
        if (v.x < minX) minX = v.x;
        if (v.y < minY) minY = v.y;
        if (v.x > maxX) maxX = v.x;
        if (v.y > maxY) maxY = v.y;
      }
      if (minX > maxX) continue; // empty ring
      faces.push({ feature, polygon, box: [minX, minY, maxX, maxY] });
    }
  }

  const parent = faces.map((_, i) => i);
  const find = (i: number): number => {
    while (parent[i] !== i) i = parent[i] = parent[parent[i]];
    return i;
  };
  const order = faces.map((_, i) => i).sort((a, b) => faces[a].box[0] - faces[b].box[0]);
  for (let oi = 0; oi < order.length; oi++) {
    const a = faces[order[oi]].box;
    for (let oj = oi + 1; oj < order.length && faces[order[oj]].box[0] <= a[2] + tolerance; oj++) {
      const b = faces[order[oj]].box;
      if (b[1] <= a[3] + tolerance && a[1] <= b[3] + tolerance) {
        parent[find(order[oi])] = find(order[oj]);
      }
    }
  }

  const parts = new globalThis.Map<number, Part>();
  for (let i = 0; i < faces.length; i++) {
    const { feature, polygon, box } = faces[i];
    const root = find(i);
    let part = parts.get(root);
    if (!part) {
      part = { members: [], minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
      parts.set(root, part);
    }
    let member = part.members.find((m) => m.feature === feature);
    if (!member) part.members.push((member = { feature, polygons: [] }));
    member.polygons.push(polygon);
    part.minX = Math.min(part.minX, box[0]);
    part.minY = Math.min(part.minY, box[1]);
    part.maxX = Math.max(part.maxX, box[2]);
    part.maxY = Math.max(part.maxY, box[3]);
  }
  return [...parts.values()];
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
