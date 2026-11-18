import { PbfReader as Pbf } from 'pbf';
import earcut from 'earcut';
import { featureFilter } from '@maplibre/maplibre-gl-style-spec';
import type { FilterSpecification, Feature, GlobalProperties } from '@maplibre/maplibre-gl-style-spec';
import { VectorTile } from './vectorTile3d';
import type { Geometry3D, VectorTileFeature } from './vectorTile3d';
import { compileColor, compileNumber } from './expression';
import type { Evaluator, Rgba, StyleValue } from './expression';

/** Flat 2D footprints to extrude into prisms alongside the 3D layer (EXPERIMENTAL). */
export interface ExtrusionRequest {
  /** Vector-tile layer with the footprint polygons (e.g. `buildings2d`). */
  sourceLayer: string;
  filter?: FilterSpecification;
  /** Roof height in metres; default `['get', 'height']`. */
  height?: StyleValue<number>;
  /** Height of the extrusion's bottom in metres; default `['get', 'min_height']`, else 0. */
  base?: StyleValue<number>;
}

/**
 * What the worker prepares besides the geometry, so that the main thread has no
 * per-vertex work when the tile arrives (see {@link Prepared}).
 */
export interface TilePrep {
  /** `fill-3d-color` / `fill-3d-opacity` as given to the layer, and what they are evaluated against. */
  color?: StyleValue<string>;
  opacity?: StyleValue<number>;
  zoom: number;
  globalState: Record<string, unknown>;
  /** The map has no terrain: the heights can be final (see Fill3DManager.bake). */
  flat: boolean;
  /** Also send the features' properties: the main thread re-evaluates the paint on a global-state change. */
  properties: boolean;
}

/**
 * Request: the fetched tile and what each layer of the source wants from it —
 * the layers of one source share a tile's fetch and its pass through the worker
 * (see TileHub). Geometry is stored tile-locally, so the scene origin is NOT needed here.
 */
export interface TileRequest {
  key: string;
  x: number;
  y: number;
  z: number;
  buffer: ArrayBuffer;
  consumers: ConsumerRequest[];
}

/** One layer's share of a {@link TileRequest}; `consumer` and `seq` come back in its results. */
export interface ConsumerRequest extends TileSpec {
  consumer: number;
  seq: number;
}

/** Which layer to read from a tile, with an optional filter, and what to build from it. */
export interface TileSpec {
  /** Vector-tile layer name to read (the `source-layer`). */
  sourceLayer: string;
  /** Optional MapLibre filter expression; features that fail are dropped. */
  filter?: FilterSpecification;
  /** Also extrude this layer's 2D footprints (EXPERIMENTAL). */
  extrusion?: ExtrusionRequest;
  /** Width in metres of the ground skirt around the building feet; 0 = none. */
  skirtRadius: number;
  /** Fake AO can be on: find inner edges and emit the AO attributes. */
  ao: boolean;
  /**
   * With `ao`, answer with the geometry alone first
   * (a {@link TileResult} without AO data and without the skirt), and send the AO
   * data in a second message ({@link AoResult}) once no tile waits for its geometry.
   * The worker keeps the parsed tile for that.
   */
  aoLater?: boolean;
  /** Metres the inner-edge darkening reaches (fill-3d-ambient-occlusion-radius): the AO grid lists edges that far. */
  aoRadius: number;
  /** What the worker prepares besides the geometry (see {@link TilePrep}). */
  prep: TilePrep;
}

/** What placing a tile on a map without terrain needs of its {@link FeatureTable}. */
export type FlatSpans = Pick<FeatureTable, 'start' | 'count' | 'rigid'>;

/** The features' vertex spans as columns ({@link FeatureSpan} without the properties): transferred, not cloned. */
export interface FeatureTable {
  start: Uint32Array;
  count: Uint32Array;
  lng: Float64Array;
  lat: Float64Array;
  /** -1: none. */
  anchor: Int32Array;
  /** NaN: none. */
  base: Float32Array;
  rigid: Uint8Array;
}

/**
 * What the worker prepared for the main thread (see {@link TilePrep}): the
 * per-vertex work of a tile's arrival. With `flat`, `position` (and `aoh`, the
 * skirt) carry the final heights for a map without terrain, and the bounding
 * spheres come along (x, y, z, radius).
 */
export interface Prepared {
  /** Per vertex: its height as emitted, relative to the feature's base (baking overwrites `position`'s). */
  baseY: Float32Array;
  /** Per vertex RGBA, 8 bits each: fill-3d-color, with a per-feature fill-3d-opacity in the alpha. */
  color: Uint8Array;
  /** Whether any vertex got an alpha below 1. */
  translucent: boolean;
  table: FeatureTable;
  /** Per feature, with {@link TilePrep.properties}. */
  properties?: Record<string, number | string | boolean>[];
  flat?: { sphere: Float32Array; skirtSphere?: Float32Array };
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
  /**
   * With terrain: the feature's group in {@link TileResult.anchors} and its base
   * height (m) in the source's height frame — the base goes to `base` plus the
   * group's terrain fit. Absent: the base goes to the terrain height at (lng, lat).
   */
  anchor?: number;
  base?: number;
  /**
   * A model built against the terrain around its anchor.
   * (lng, lat) is that anchor, and the whole feature goes to the terrain height
   * there as one rigid body (see Fill3DManager.bake).
   */
  rigid?: boolean;
}

/**
 * Per group of touching buildings its wall feet standing on the ground — what
 * the main thread fits to the terrain to place the whole group (see
 * Fill3DManager.bake). Group `g` owns the feet `start[g]` to `start[g + 1]`.
 */
export interface TerrainAnchors {
  start: Uint32Array;
  /** Per foot: its ground column (see {@link TileResult.columnXZ}). */
  column: Uint32Array;
  /** Per foot: its height (m) in the source's height frame. */
  height: Float32Array;
  /** Per foot: the length of wall (m) it stands for. */
  weight: Float32Array;
}

/** Ground skirt: a strip along the foot of every wall that stands on the ground. */
export interface SkirtResult {
  /** xyz per vertex in tile-local metres; y = 0 (the ground height is baked in later). */
  position: Float32Array;
  /**
   * Per vertex, 2 normalized bytes, in skirt radii: (u, v) — beyond a free end
   * of the strip (0 along it) and out from the wall foot (0 on the foot, 1 on
   * the outer edge). Both are linear across the strip, so length(u, v) is the
   * exact distance from the foot per fragment, rounded beyond free ends.
   */
  coord: Uint8Array;
  index: Uint16Array | Uint32Array;
  /** Per vertex: its ground column (see {@link TileResult.columnXZ}). */
  column: Uint32Array;
}

export interface TileResult {
  key: string;
  /** The request's {@link ConsumerRequest.consumer} and {@link ConsumerRequest.seq}. */
  consumer?: number;
  seq?: number;
  /** From the worker, for a tile with geometry; `features` is empty then (see {@link Prepared.table}). */
  prepared?: Prepared;
  /**
   * xyz per vertex in TRUE METRES relative to the tile's NW corner:
   * x=east, y=height-above-base, z=north. Scene-origin independent.
   */
  position: Float32Array;
  /** Per-vertex face normal in the same (east, up, north) frame, scaled to ±127
   *  (a normalized Int8 attribute — a quarter of the bytes of float32). */
  normal: Int8Array;
  /**
   * Fake ambient occlusion at the wall foot, 1 per vertex, decimetres: the
   * height above the ground (here: above the base; the main thread re-bakes it
   * against the terrain). Empty without AO.
   */
  aoh: Int16Array;
  /**
   * Fake AO at inner edges: per vertex its face's AO table entry
   * ({@link AO_NO_FACE}: none) and its position in the face's plane (2 per
   * vertex, units of {@link AO_UV_UNIT_M}, around the face's centre). Linear in
   * the face, so the fragment gets its exact plane position and looks up the
   * face's inner edges near it in {@link aoTables}. Empty without AO.
   */
  aoFace: Uint16Array;
  aoUV: Int16Array;
  aoTables?: AoTables;
  /**
   * Per vertex: its ground column — an index into `columnXZ`, the distinct
   * (x, z) positions of the tile's vertices and skirt vertices. A wall's top
   * and bottom, the roof above it and the skirt at its foot share columns, so
   * baking samples the terrain once per column instead of once per vertex.
   */
  column: Uint16Array | Uint32Array;
  /** x, z (tile-local metres) per column. */
  columnXZ: Float32Array;
  /** Uint16 while the tile has at most 65536 vertices (half the bytes), else Uint32. */
  index: Uint16Array | Uint32Array;
  features: FeatureSpan[];
  /** Absent when no building of the tile has a base height in the source's frame. */
  anchors?: TerrainAnchors;
  /** meterInMercatorCoordinateUnits at the tile's latitude (for placement). */
  tileScale: number;
  skirt?: SkirtResult;
}

/**
 * The AO data of a tile whose geometry was sent
 * before — per vertex of that geometry (same vertices, same order), plus the
 * tables, the skirt and the columns including the skirt's.
 */
export interface AoResult {
  key: string;
  consumer?: number;
  seq?: number;
  phase: 'ao';
  /** With {@link TilePrep.flat}: per vertex its height above the ground (dm), final for a map without terrain. */
  aoh?: Int16Array;
  /** With {@link TilePrep.flat}: the skirt's bounding sphere (its heights are final too). */
  skirtSphere?: Float32Array;
  aoFace: Uint16Array;
  aoUV: Int16Array;
  aoTables?: AoTables;
  columnXZ: Float32Array;
  skirt?: SkirtResult;
}

/** `aoFace` of a face without inner edges. */
export const AO_NO_FACE = 65535;
/** Unit (m) of `aoUV`: Int16 reaches ±655 m from a face's centre. */
export const AO_UV_UNIT_M = 0.02;
/** Cell size (m) of a face's grid in the AO table. */
export const AO_CELL_M = 4;
/** Row width (texels) of the AO table textures: 2^AO_TEXTURE_SHIFT. */
export const AO_TEXTURE_SHIFT = 9;
export const AO_TEXTURE_WIDTH = 1 << AO_TEXTURE_SHIFT;
/** Zones are listed in the cells they reach this much (m) beyond — robust at cell borders. */
const AO_CELL_MARGIN_M = 0.1;
/** Faces with at most this many inner edges get no grid: the fragment checks them all, straight from the edge table. */
const AO_DIRECT_EDGES = 4;
/** At most this many inner edges are listed per grid cell (the 8 bits of a cell's count). */
const AO_MAX_PER_CELL = 255;

/**
 * The inner edges of a tile's faces for the fake AO, as four textures' data
 * (rows of {@link AO_TEXTURE_WIDTH} texels). The fragment reads its face's
 * entry, the cell of the face's grid it lies in, and darkens by the strongest
 * of the cell's edges — any number of edges per face, each evaluated exactly.
 */
export interface AoTables {
  /**
   * Per face 4 × uint32 (RGBA32UI): grid origin a, b (two Int16 decimetres in
   * the plane), nx + 65536·ny, first cell, first edge. A face with few edges
   * has no grid: nx = 0, ny = its edge count.
   */
  faces: Uint32Array;
  /** Per grid cell 1 × uint32 (R32UI): first list entry · 256 + entry count. */
  cells: Uint32Array;
  /** Edge indices relative to the face's first edge (R16UI), the entries of a cell consecutive. */
  lists: Uint16Array;
  /**
   * Per edge 4 × uint32 (RGBA32UI), plane metres: start a, b (float bits);
   * unit direction as two snorm16 (packSnorm2x16); half floats of the length —
   * its sign the side the face lies on (left of the direction: +) — and the
   * strength (1 + cos of the opening angle).
   */
  edges: Uint32Array;
}

/** Two values in [-1, 1] as GLSL packSnorm2x16 does it (first in the low 16 bits). */
function snorm2x16(a: number, b: number): number {
  const s = (v: number) => Math.round(Math.max(-1, Math.min(1, v)) * 32767) & 0xffff;
  return (s(a) | (s(b) << 16)) >>> 0;
}

/** Float32 → IEEE half (bits), rounded to nearest. */
const HALF_F32 = new Float32Array(1);
const HALF_U32 = new Uint32Array(HALF_F32.buffer);
function toHalf(v: number): number {
  HALF_F32[0] = v;
  const x = HALF_U32[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = ((x >>> 23) & 0xff) - 127 + 15;
  const mant = x & 0x7fffff;
  if (exp >= 31) return sign | 0x7bff; // clamp to the largest finite half
  if (exp <= 0) return sign | Math.round((mant | 0x800000) / 2 ** (14 - exp)); // subnormal
  const h = sign | (exp << 10) | Math.round(mant / 8192);
  return h;
}

/** A face's inner edge in its plane: start, unit direction, length, side (±1), strength. */
interface PlaneEdge {
  a: number;
  b: number;
  ua: number;
  ub: number;
  len: number;
  side: number;
  k: number;
}

/** Collects the AO tables of a tile, face by face. */
class AoTableBuilder {
  faces: number[] = [];
  cells: number[] = [];
  lists: number[] = [];
  edges: number[] = [];
  private readonly f32 = new Float32Array(1);
  private readonly u32 = new Uint32Array(this.f32.buffer);

  /** Add a face's edges with its plane box [a0, b0, a1, b1]; returns its index (AO_NO_FACE when full). */
  add(edges: PlaneEdge[], box: [number, number, number, number], radius: number): number {
    const index = this.faces.length / 4;
    if (index >= AO_NO_FACE) return AO_NO_FACE;
    const firstEdge = this.edges.length / 4;
    for (const e of edges) this.edges.push(this.bits(e.a), this.bits(e.b), snorm2x16(e.ua, e.ub), (toHalf(e.len * e.side) | (toHalf(e.k) << 16)) >>> 0);
    if (edges.length <= AO_DIRECT_EDGES) {
      // No grid: the fragment checks them all, straight from the edge table.
      this.faces.push(0, 65536 * edges.length, 0, firstEdge);
      return index;
    }
    const firstCell = this.cells.length;
    // The grid origin in whole decimetres (two Int16 in the face entry).
    const a0 = Math.floor(box[0] * 10) / 10;
    const b0 = Math.floor(box[1] * 10) / 10;
    const nx = Math.max(1, Math.ceil((box[2] - a0) / AO_CELL_M));
    const ny = Math.max(1, Math.ceil((box[3] - b0) / AO_CELL_M));
    const lists: number[][] = [];
    for (let c = 0; c < nx * ny; c++) lists.push([]);
    const h = AO_CELL_M / 2;
    edges.forEach((e, i) => {
      // Its zone (where it can darken): along [-radius, len + radius], up to
      // radius into the face; a little more, so that a fragment on a cell
      // border still finds it.
      const na = -e.ub * e.side;
      const nb = e.ua * e.side;
      const hu = e.len / 2 + radius + AO_CELL_MARGIN_M;
      const hn = radius / 2 + AO_CELL_MARGIN_M;
      const ca = e.a + e.ua * (e.len / 2) + na * (radius / 2);
      const cb = e.b + e.ub * (e.len / 2) + nb * (radius / 2);
      const ra = hu * Math.abs(e.ua) + hn * Math.abs(na);
      const rb = hu * Math.abs(e.ub) + hn * Math.abs(nb);
      const x0 = Math.max(0, Math.floor((ca - ra - a0) / AO_CELL_M));
      const x1 = Math.min(nx - 1, Math.floor((ca + ra - a0) / AO_CELL_M));
      const y0 = Math.max(0, Math.floor((cb - rb - b0) / AO_CELL_M));
      const y1 = Math.min(ny - 1, Math.floor((cb + rb - b0) / AO_CELL_M));
      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          // Separating axes between the zone and the cell: the cell's two, the zone's two.
          const da = a0 + (x + 0.5) * AO_CELL_M - ca;
          const db = b0 + (y + 0.5) * AO_CELL_M - cb;
          if (Math.abs(da) > h + ra || Math.abs(db) > h + rb) continue;
          if (Math.abs(da * e.ua + db * e.ub) > hu + h * (Math.abs(e.ua) + Math.abs(e.ub))) continue;
          if (Math.abs(da * na + db * nb) > hn + h * (Math.abs(na) + Math.abs(nb))) continue;
          lists[y * nx + x].push(i);
        }
      }
    });
    for (const list of lists) {
      const n = Math.min(list.length, AO_MAX_PER_CELL);
      this.cells.push(this.lists.length * 256 + n);
      for (let k = 0; k < n; k++) this.lists.push(list[k]);
    }
    const dm16 = (v: number) => Math.round(v * 10) & 0xffff;
    this.faces.push((dm16(a0) | (dm16(b0) << 16)) >>> 0, nx + 65536 * ny, firstCell, firstEdge);
    return index;
  }

  private bits(v: number): number {
    this.f32[0] = v;
    return this.u32[0];
  }

  /** The tables, each padded to whole texture rows. */
  pack(): AoTables {
    const rows = (n: number, channels: number) => Math.max(1, Math.ceil(n / channels / AO_TEXTURE_WIDTH)) * AO_TEXTURE_WIDTH * channels;
    const fill = <T extends Uint32Array | Uint16Array>(out: T, a: number[]) => {
      out.set(a);
      return out;
    };
    return {
      faces: fill(new Uint32Array(rows(this.faces.length, 4)), this.faces),
      cells: fill(new Uint32Array(rows(this.cells.length, 1)), this.cells),
      lists: fill(new Uint16Array(rows(this.lists.length, 1)), this.lists),
      edges: fill(new Uint32Array(rows(this.edges.length, 4)), this.edges),
    };
  }
}

type Vec3 = [number, number, number];

/** One planar face of a building: its rings (outer first), their vertices one after the other. */
interface Face {
  /**
   * x, y, z per vertex — east, up (relative to the base), north, in metres — of
   * all faces read from one feature: a tile has millions of vertices, and an
   * object for each costs more than all that is done with it. The face's are
   * the `count` vertices from offset `at`.
   */
  xyz: Float64Array;
  at: number;
  count: number;
  /** The vertex each ring after the outer one starts at, counted within the face (as earcut takes them). */
  holes: number[];
  /**
   * Height frame the face's edges are matched in: the 3D source's own heights
   * (0) or, for extrusions, heights above their ground (1) — the two don't mix.
   */
  frame: number;
  /** Add to a vertex's base-relative y to get its height in that frame. */
  yOff: number;
  /** Newell normal — points INTO the building in this frame (see the shader). */
  n: Vec3;
  /** Stretches of the face's edges it shares as an inner edge with another face. */
  inner: InnerStretch[];
}

/** A stretch of the face's boundary along which it meets another face in an inner (concave) edge. */
interface InnerStretch {
  /** Stretches under one key lie on one line and get merged. */
  key: object;
  /** The line (frame coordinates, see {@link EdgeRef}) and the interval along it. */
  origin: Vec3;
  dir: Vec3;
  lo: number;
  hi: number;
  /** Unit direction within the face, from the line towards the darkened side. */
  m: Vec3;
  /** Strength: 1 + cos of the opening angle between the two faces (see {@link TileResult.aok}). */
  k: number;
}

/** A face edge, positioned on its line in the face's height frame. */
interface EdgeRef {
  f: Face;
  /** Its two vertices: offsets in the face's `xyz`. */
  a: number;
  b: number;
  /** Unit direction of the line (canonical sign) and the parameter range the edge covers. */
  dir: Vec3;
  t0: number;
  t1: number;
  /** Point on the line at parameter 0, in the frame (y includes yOff). */
  origin: Vec3;
}

/** Fill-in timings and counts for benchmarks (not used in rendering). */
export interface BuildStats {
  parseMs: number;
  innerMs: number;
  emitMs: number;
  extrusionMs: number;
  skirtMs: number;
  /** Pairs of faces found to meet in an inner edge. */
  innerPairs: number;
}

/** A building's faces, grouped by the source feature they style as. */
interface Member {
  properties: Record<string, number | string | boolean>;
  faces: Face[];
}

/** A building, anchored at (lng, lat) or — see {@link terrainAnchors} — with its group. */
interface Building {
  members: Member[];
  lng: number;
  lat: number;
  anchor?: number;
  base?: number;
  /** Placed rigidly at (lng, lat), see {@link FeatureSpan.rigid}. */
  rigid?: boolean;
}

/** A wall foot on the ground, in tile-local metres, with the wall's outward direction. */
interface GroundEdge {
  ax: number;
  az: number;
  bx: number;
  bz: number;
  nx: number;
  nz: number;
}

/** One 3D feature of the tile; the first polygon of its geometry is its bounding box. */
interface SourceFeature {
  properties: Record<string, number | string | boolean>;
  extent: number;
  baseZ: number;
  geometry: Geometry3D;
}

/** A connected building: its faces (polygons of the feature's geometry), per source feature, and their extent. */
interface Part {
  members: { feature: SourceFeature; polygons: number[] }[];
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

/** Vertices this close (metres) above a building's base count as its foot. */
const BASE_EPSILON_M = 0.05;

/** A wall whose foot is at most this far (metres) above its building's base
 *  stands on the ground; a higher one stands on a roof. */
const GROUND_FOOT_MAX_M = 8;

/** Extruded walls standing on the ground reach this far below it (metres), as
 *  a reserve for slopes; the terrain hides it, the flat map clamps it to 0. */
const EXTRUSION_GROUND_DEPTH_M = 2;

/** Faces whose normals are closer than this (cosine, ~10°) count as one surface: no inner edge. */
const COPLANAR_COS = 0.985;

/** Each face must run off at least this much (cosine) to the other's front for an inner edge. */
const CONCAVE_EPSILON = 0.02;

/** Edges on one line must overlap by at least this much (metres) to count as shared. */
const MIN_SHARED_M = 0.2;

/** Stretches on the same face edge closer than this (metres) merge into one. */
const MERGE_GAP_M = 0.1;

/** Skirt strips are cut into pieces no longer than this (metres) to follow the terrain. */
const SKIRT_STEP_M = 12;

/** A skirt corner's mitre may reach at most this many radii out. */
const SKIRT_MITRE_LIMIT = 4;

/** Feet turning by less than this (cosine, ~0.6°) continue one straight skirt
 *  piece: off by at most ~10 cm over 10 m. */
const SKIRT_STRAIGHT_COS = 0.99995;

// MapLibre's mercator metric (earthRadius 6371008.8 m).
const EARTH_CIRCUMFERENCE = 2 * Math.PI * 6371008.8;

/** meterInMercatorCoordinateUnits() equivalent for a given latitude. */
function mercatorScaleAtLat(latDeg: number): number {
  return 1 / (EARTH_CIRCUMFERENCE * Math.cos((latDeg * Math.PI) / 180));
}

/** The fake-AO attributes of a tile's vertices, and the tables they point into. */
class AoAttributes {
  /** Per vertex: {@link TileResult.aoFace} and {@link TileResult.aoUV}; no face and (0, 0) until a face with inner edges sets them. */
  readonly face: Uint16Array;
  readonly uv: Int16Array;
  readonly tables = new AoTableBuilder();

  /** For `vertices` vertices, with inner edges reaching `radius` metres. */
  constructor(
    vertices: number,
    readonly radius: number,
  ) {
    this.face = new Uint16Array(vertices).fill(AO_NO_FACE);
    this.uv = new Int16Array(vertices * 2);
  }
}

/**
 * The tile's ground columns: the distinct (x, z) of its vertices and skirt
 * vertices, numbered in the order they are first asked for.
 */
class Columns {
  /** x, z per column. */
  xz: number[] = [];
  // The columns by position in whole millimetres, in an open-addressing table
  // (x, z, column per slot; -1: free): a tile asks millions of times, and a Map
  // keyed by a number beyond 2^31 costs an allocation per question.
  private slotX = new Int32Array(4096);
  private slotZ = new Int32Array(4096);
  private slotColumn = new Int32Array(4096).fill(-1);

  /** The ground column at (x, z), positions within a millimetre sharing one. */
  at(x: number, z: number): number {
    const xi = Math.round(x * 1000) | 0;
    const zi = Math.round(z * 1000) | 0;
    let mask = this.slotColumn.length - 1;
    let h = (Math.imul(xi, 0x9e3779b1) ^ Math.imul(zi, 0x85ebca6b)) >>> 0;
    h = (h ^ (h >>> 15)) & mask;
    for (;;) {
      const c = this.slotColumn[h];
      if (c === -1) break;
      if (this.slotX[h] === xi && this.slotZ[h] === zi) return c;
      h = (h + 1) & mask;
    }
    const c = this.xz.length / 2;
    this.xz.push(x, z);
    this.slotX[h] = xi;
    this.slotZ[h] = zi;
    this.slotColumn[h] = c;
    if ((c + 1) * 2 > this.slotColumn.length) this.growSlots();
    return c;
  }

  /** Double the table and put the columns into it again. */
  private growSlots() {
    const { slotX, slotZ, slotColumn } = this;
    const size = slotColumn.length * 2;
    const mask = size - 1;
    this.slotX = new Int32Array(size);
    this.slotZ = new Int32Array(size);
    this.slotColumn = new Int32Array(size).fill(-1);
    for (let i = 0; i < slotColumn.length; i++) {
      if (slotColumn[i] === -1) continue;
      let h = (Math.imul(slotX[i], 0x9e3779b1) ^ Math.imul(slotZ[i], 0x85ebca6b)) >>> 0;
      h = (h ^ (h >>> 15)) & mask;
      while (this.slotColumn[h] !== -1) h = (h + 1) & mask;
      this.slotX[h] = slotX[i];
      this.slotZ[h] = slotZ[i];
      this.slotColumn[h] = slotColumn[i];
    }
  }
}

/**
 * Accumulates the tile's mesh, vertex by vertex — straight into the arrays the
 * result is made of: the number of vertices is known beforehand.
 */
class MeshBuilder {
  /** Vertices emitted so far. */
  vertexCount = 0;
  readonly position: Float32Array;
  readonly normal: Int8Array;
  readonly aoh: Int16Array;
  /** The triangles' vertices, `indexCount` of them so far. */
  index: Uint32Array;
  indexCount = 0;
  /** Per vertex its ground column. */
  readonly column: Uint32Array;
  readonly columns = new Columns();
  features: FeatureSpan[] = [];
  ground: GroundEdge[] = [];

  /** For `vertices` vertices; with `ao`, their fake-AO attributes are emitted too. */
  constructor(
    vertices: number,
    readonly ao?: AoAttributes,
  ) {
    this.position = new Float32Array(vertices * 3);
    this.normal = new Int8Array(vertices * 3);
    this.aoh = new Int16Array(ao ? vertices : 0);
    // A face of n vertices has n - 2 triangles, with holes two more for each.
    this.index = new Uint32Array(vertices * 3);
    this.column = new Uint32Array(vertices);
  }

  /** Append a face's triangles (indices into the face's vertices, the first of which is `base`). */
  addTriangles(tri: number[], base: number) {
    if (this.indexCount + tri.length > this.index.length) {
      const grown = new Uint32Array(Math.max(this.index.length * 2, this.indexCount + tri.length));
      grown.set(this.index);
      this.index = grown;
    }
    for (let t = 0; t < tri.length; t++) this.index[this.indexCount++] = tri[t] + base;
  }
}

/**
 * A tile's buildings, parsed. {@link emitTile} turns them into geometry; kept by
 * the worker, {@link emitAo} later derives the AO data from the same faces
 * (fake AO in two passes).
 */
export interface ParsedTile {
  buildings: Building[];
  /** meterInMercatorCoordinateUnits at the tile's latitude. */
  tileScale: number;
  /** The inner edges are on the faces already (findInnerEdges adds to them). */
  innerFound?: boolean;
  /**
   * Left by {@link emitTile} for {@link emitAo}, which then need not walk the
   * vertices for them again: per vertex its ground column, the columns
   * themselves (the skirt adds its own) and the wall feet.
   */
  forAo?: { column: Uint16Array | Uint32Array; columns: Columns; ground: GroundEdge[] };
}

function lapOf(stats?: BuildStats) {
  let t = performance.now();
  return (key: Exclude<keyof BuildStats, 'innerPairs'>) => {
    const now = performance.now();
    if (stats) stats[key] += now - t;
    t = now;
  };
}

/**
 * A fetched tile, opened once for all layers that read it: each of a layer's
 * features is decoded (its properties) only once, however many layers filter it.
 */
export interface TileReader {
  tile: VectorTile;
  featuresOf(sourceLayer: string): VectorTileFeature[];
}

export function readTile(buffer: ArrayBuffer): TileReader {
  const tile = new VectorTile(new Pbf(new Uint8Array(buffer)));
  const decoded = new globalThis.Map<string, VectorTileFeature[]>();
  return {
    tile,
    featuresOf(sourceLayer) {
      let features = decoded.get(sourceLayer);
      if (!features) {
        const layer = tile.layers[sourceLayer];
        features = [];
        if (layer) for (let i = 0; i < layer.length; i++) features.push(layer.feature(i));
        decoded.set(sourceLayer, features);
      }
      return features;
    },
  };
}

/** Parse one tile's 3D-feature layer (and the footprints to extrude) into buildings of faces, tile-local metres. */
export function parseTile(
  tx: number,
  ty: number,
  tz: number,
  source: ArrayBuffer | TileReader,
  sourceLayer: string,
  filter?: FilterSpecification,
  extrusion?: ExtrusionRequest,
  stats?: BuildStats,
): ParsedTile {
  const lap = lapOf(stats);
  const pow = 2 ** tz;
  // Metric at the tile's centre latitude (geometry is stored relative to the corner).
  const tileLat = mercToLngLat((tx + 0.5) / pow, (ty + 0.5) / pow)[1];
  const tileScale = mercatorScaleAtLat(tileLat);

  // All buildings of the tile are collected first: inner edges are searched
  // tile-wide, also between neighbouring buildings.
  const buildings: Building[] = [];
  const reader = source instanceof ArrayBuffer ? readTile(source) : source;
  const tile = reader.tile;
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
    const features = reader.featuresOf(sourceLayer);
    for (let i = 0; i < features.length; i++) {
      const feature = features[i];
      if (feature.type !== 0) continue; // only the custom 3D geometry
      if (ff && !ff.filter(globals, feature as unknown as Feature)) continue;

      const geometry = feature.loadGeometry();
      if (geometry.polygonEnd.length < 2) continue; // need bbox + at least one real polygon

      // First polygon is a bounding box: [min, max] corners → base z.
      const baseZ = geometry.xyz[2];
      const id = feature.properties.id ?? `\u0000${i}`; // no id: a group of its own
      let group = byId.get(id);
      if (!group) byId.set(id, (group = []));
      group.push({ properties: feature.properties, extent: feature.extent, baseZ, geometry });
    }

    for (const group of byId.values()) {
      const extent = group[0].extent;
      const tolerance = PART_GAP_M * extent * pow * tileScale; // metres → extent units
      // A landmark is one rigid body at its anchor, so its
      // parts need not be told apart — and it is one id with tens of thousands of
      // faces, for which the search for connected parts is not made (Vienna's
      // Hofburg, 78,000 faces: 1.2 of 1.8 s). It is drawn in the tile that holds
      // the centre of its bounding box.
      const tags0 = group[0].properties;
      const landmark = typeof tags0.anchor_lng === 'number' && typeof tags0.anchor_lat === 'number';
      for (const part of landmark ? [wholePart(group)] : connectedParts(group, tolerance)) {
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

        // A feature carrying `anchor_lng` / `anchor_lat` is a
        // model built against the terrain around that anchor (a landmark from the
        // generator). Every part of it is placed at the anchor, not at its own centre.
        const rigid = landmark;
        const [lng, lat] = rigid
          ? [tags0.anchor_lng as number, tags0.anchor_lat as number]
          : mercToLngLat((tx + cx / extent) / pow, (ty + cy / extent) / pow);
        const members: Member[] = part.members.map(({ feature: src, polygons }) => ({
          properties: src.properties,
          faces: polygons.map((polygon) => sourceFace(src, polygon, pow, tileScale)),
        }));
        buildings.push(rigid ? { members, lng, lat, rigid } : { members, lng, lat });
      }
    }
  }
  lap('parseMs');

  if (extrusion) addExtrusions(buildings, tile, extrusion, tx, ty, tz, pow, tileScale);
  lap('extrusionMs');
  return { buildings, tileScale };
}

/** The number of vertices the buildings' faces have. */
function countVertices(buildings: Building[]): number {
  let count = 0;
  for (const b of buildings) for (const m of b.members) for (const f of m.faces) count += f.count;
  return count;
}

/** Find the tile's inner edges once (they are added to the faces). */
function findInner(parsed: ParsedTile, stats?: BuildStats) {
  if (parsed.innerFound) return;
  const pairs = findInnerEdges(parsed.buildings.flatMap((b) => b.members.flatMap((m) => m.faces)));
  parsed.innerFound = true;
  if (stats) stats.innerPairs += pairs;
}

/** Triangulate a parsed tile into tile-local metre geometry — with `ao` including the AO data, with `skirtRadius` the skirt. */
export function emitTile(parsed: ParsedTile, skirtRadius = 0, ao = true, stats?: BuildStats, aoRadius = 3): Omit<TileResult, 'key'> {
  const lap = lapOf(stats);
  const { buildings, tileScale } = parsed;
  // Without AO there is nothing to find, and no AO attributes to emit.
  if (ao) findInner(parsed, stats);
  lap('innerMs');

  const vertices = countVertices(buildings);
  const out = new MeshBuilder(vertices, ao ? new AoAttributes(vertices, aoRadius) : undefined);
  const anchors = terrainAnchors(buildings, out.columns);
  for (const b of buildings) emitBuilding(out, b);
  const index = out.index.subarray(0, out.indexCount);
  const result = {
    position: out.position,
    normal: out.normal,
    aoh: out.aoh,
    aoFace: out.ao ? out.ao.face : new Uint16Array(0),
    aoUV: out.ao ? out.ao.uv : new Int16Array(0),
    aoTables: out.ao?.tables.pack(),
    index: out.vertexCount <= 65536 ? new Uint16Array(index) : index.slice(),
    // 16 bit while the tile has at most 65536 columns (half the bytes), like the index.
    column: out.columns.xz.length / 2 <= 65536 ? new Uint16Array(out.column) : out.column,
    features: out.features,
    anchors,
    tileScale,
  };
  lap('emitMs');
  const skirt = skirtRadius > 0 && out.ground.length ? buildSkirt(out.ground, skirtRadius, out.columns) : undefined;
  lap('skirtMs');
  // The columns as the vertices alone numbered them: where an AO pass takes up
  // (`column` as a copy — the result's is given away).
  if (!ao && !skirt) parsed.forAo = { column: result.column.slice(), columns: out.columns, ground: out.ground };
  // After the skirt: its vertices add columns too.
  return { ...result, columnXZ: new Float32Array(out.columns.xz), skirt };
}

/**
 * The AO data of a tile whose geometry
 * {@link emitTile} gave out before (without AO) — per vertex of that geometry,
 * in its order — plus the tables, the skirt and the columns including the
 * skirt's. Walks the faces as emitTile does (same vertices), but only those
 * with inner edges; the columns and the wall feet are the ones emitTile left.
 */
export function emitAo(
  parsed: ParsedTile,
  skirtRadius: number,
  aoRadius: number,
  stats?: BuildStats,
): Omit<AoResult, 'key' | 'phase'> & { column: Uint16Array | Uint32Array } {
  const lap = lapOf(stats);
  findInner(parsed, stats);
  lap('innerMs');
  if (!parsed.forAo) emitTile(parsed, 0, false);
  const { column, columns, ground } = parsed.forAo!;
  // The skirt changes them: they are good for one AO pass.
  parsed.forAo = undefined;
  const ao = new AoAttributes(countVertices(parsed.buildings), aoRadius);
  let first = 0;
  for (const b of parsed.buildings) {
    for (const m of b.members) {
      for (const f of m.faces) {
        emitFaceAo(ao, f, first);
        first += f.count;
      }
    }
  }
  const result = { aoFace: ao.face, aoUV: ao.uv, aoTables: ao.tables.pack() };
  lap('emitMs');
  const skirt = skirtRadius > 0 && ground.length ? buildSkirt(ground, skirtRadius, columns) : undefined;
  lap('skirtMs');
  // `column`: per vertex its ground column, as in emitTile's result (not sent to the main thread again).
  return { ...result, columnXZ: new Float32Array(columns.xz), skirt, column };
}

// ------------------------------------------------------------------ prepare

/** The skirt floats this far (metres) above the ground, against z-fighting with the terrain. */
export const SKIRT_LIFT_M = 0.4;

const NEUTRAL_COLOR: Rgba = [1, 1, 1, 1];
/** Compiled paint, per request text: a layer sends the same paint with every tile. */
const paints = new globalThis.Map<string, { color: Evaluator<Rgba>; opacity: Evaluator<number> }>();
function paintOf(prep: TilePrep) {
  const key = JSON.stringify([prep.color, prep.opacity]);
  let paint = paints.get(key);
  if (!paint) {
    paint = {
      color: compileColor(prep.color, 'fill-3d-color', NEUTRAL_COLOR, 3),
      opacity: compileNumber(prep.opacity, 'fill-3d-opacity', 1, 3),
    };
    paints.set(key, paint);
  }
  return paint;
}

export function featureTable(features: FeatureSpan[]): FeatureTable {
  const n = features.length;
  const table: FeatureTable = {
    start: new Uint32Array(n),
    count: new Uint32Array(n),
    lng: new Float64Array(n),
    lat: new Float64Array(n),
    anchor: new Int32Array(n),
    base: new Float32Array(n),
    rigid: new Uint8Array(n),
  };
  for (let i = 0; i < n; i++) {
    const b = features[i];
    table.start[i] = b.start;
    table.count[i] = b.count;
    table.lng[i] = b.lng;
    table.lat[i] = b.lat;
    table.anchor[i] = b.anchor ?? -1;
    table.base[i] = b.base ?? NaN;
    table.rigid[i] = b.rigid ? 1 : 0;
  }
  return table;
}

/**
 * Per-vertex RGBA of a tile: fill-3d-color per feature, with a per-feature
 * fill-3d-opacity in the alpha (a feature-constant opacity is the layer's
 * `u_opacity` uniform instead). Colours stay sRGB — the shader shades and
 * outputs in sRGB, like fill-extrusion.
 */
export function vertexColors(
  vertexCount: number,
  table: FeatureTable,
  properties: Record<string, unknown>[],
  color: Evaluator<Rgba>,
  opacity: Evaluator<number>,
  zoom: number,
  state: Record<string, unknown>,
): { color: Uint8Array; translucent: boolean } {
  const rgba = new Uint8Array(vertexCount * 4);
  let translucent = false;
  for (let i = 0; i < table.start.length; i++) {
    const [r, g, bl, ca] = color(zoom, properties[i], undefined, state);
    const a = opacity.featureConstant ? ca : ca * opacity(zoom, properties[i], undefined, state);
    const R = Math.round(r * 255), G = Math.round(g * 255), B = Math.round(bl * 255), A = Math.round(a * 255);
    if (A < 255) translucent = true;
    for (let v = table.start[i], end = v + table.count[i]; v < end; v++) {
      rgba[v * 4] = R;
      rgba[v * 4 + 1] = G;
      rgba[v * 4 + 2] = B;
      rgba[v * 4 + 3] = A;
    }
  }
  return { color: rgba, translucent };
}

/**
 * The flat map's ground under the vertices of rigid models,
 * per ground column, relative to the map: the map itself, or the model's lowest
 * point in that column where it reaches below the map (a wall foot on lower
 * ground shows through the flat map, and the wall darkens towards that foot).
 * Undefined without rigid models.
 */
export function flatGroundOfRigid(table: FlatSpans, column: ArrayLike<number>, baseY: Float32Array, columns: number): Float32Array | undefined {
  let ground: Float32Array | undefined;
  for (let i = 0; i < table.start.length; i++) {
    if (!table.rigid[i]) continue;
    ground ??= new Float32Array(columns); // 0: the map
    for (let v = table.start[i], end = v + table.count[i]; v < end; v++) {
      const c = column[v];
      if (baseY[v] < ground[c]) ground[c] = baseY[v];
    }
  }
  return ground;
}

/**
 * A map without terrain: the heights as Fill3DManager.bake gives them there.
 * `position` gets its final y (ground walls come up to the map, a rigid model
 * stays as built), `aoh` each vertex's height above the ground in decimetres.
 */
export function bakeFlat(
  position: Float32Array | undefined,
  aoh: Int16Array | undefined,
  baseY: Float32Array,
  table: FlatSpans,
  column: ArrayLike<number>,
  columns: number,
) {
  const rigidGround = aoh ? flatGroundOfRigid(table, column, baseY, columns) : undefined;
  for (let i = 0; i < table.start.length; i++) {
    const rigid = table.rigid[i] === 1;
    for (let v = table.start[i], end = v + table.count[i]; v < end; v++) {
      const rel = baseY[v];
      const y = rigid ? rel : Math.max(rel, 0);
      if (position) position[v * 3 + 1] = y;
      if (!aoh) continue;
      const aboveGround = rigid ? (rigidGround ? rel - rigidGround[column[v]] : rel) : y;
      aoh[v] = Math.max(-32766, Math.min(32766, Math.round(aboveGround * 10)));
    }
  }
}

/** A bounding sphere as three.js computes it: around the bounding box's centre. x, y, z, radius. */
export function boundingSphere(position: Float32Array): Float32Array {
  let minX = Infinity, minY = Infinity, minZ = Infinity, maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < position.length; i += 3) {
    const x = position[i], y = position[i + 1], z = position[i + 2];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }
  if (minX > maxX) return new Float32Array(4);
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2, cz = (minZ + maxZ) / 2;
  let r2 = 0;
  for (let i = 0; i < position.length; i += 3) {
    const dx = position[i] - cx, dy = position[i + 1] - cy, dz = position[i + 2] - cz;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 > r2) r2 = d2;
  }
  return new Float32Array([cx, cy, cz, Math.sqrt(r2)]);
}

/** A skirt on the flat map: its heights and its bounding sphere. */
export function flatSkirt(skirt: SkirtResult): Float32Array {
  for (let i = 1; i < skirt.position.length; i += 3) skirt.position[i] = SKIRT_LIFT_M;
  return boundingSphere(skirt.position);
}

/**
 * Do on the worker what the main thread did per vertex when a tile arrived
 * (see {@link TilePrep}); changes `result` in place: its `features` are emptied
 * (the spans go as columns), and with `prep.flat` its heights are final.
 */
export function prepare(result: Omit<TileResult, 'key'>, prep: TilePrep): Prepared {
  const vertexCount = result.position.length / 3;
  const table = featureTable(result.features);
  const properties = result.features.map((f) => f.properties);
  const baseY = new Float32Array(vertexCount);
  for (let v = 0; v < vertexCount; v++) baseY[v] = result.position[v * 3 + 1];
  const paint = paintOf(prep);
  const colors = vertexColors(vertexCount, table, properties, paint.color, paint.opacity, prep.zoom, prep.globalState);
  const prepared: Prepared = { baseY, color: colors.color, translucent: colors.translucent, table };
  if (prep.properties) prepared.properties = properties;
  if (prep.flat && vertexCount) {
    bakeFlat(result.position, result.aoh.length ? result.aoh : undefined, baseY, table, result.column, result.columnXZ.length / 2);
    prepared.flat = { sphere: boundingSphere(result.position) };
    if (result.skirt) prepared.flat.skirtSphere = flatSkirt(result.skirt);
  }
  result.features = [];
  return prepared;
}

/** Parse + triangulate one tile's 3D-feature layer into tile-local metre geometry. */
export function buildTile(
  tx: number,
  ty: number,
  tz: number,
  buffer: ArrayBuffer,
  sourceLayer: string,
  filter?: FilterSpecification,
  extrusion?: ExtrusionRequest,
  skirtRadius = 0,
  ao = true,
  stats?: BuildStats,
  aoRadius = 3,
): Omit<TileResult, 'key'> {
  return emitTile(parseTile(tx, ty, tz, buffer, sourceLayer, filter, extrusion, stats), skirtRadius, ao, stats, aoRadius);
}

/** `holes` of a face without any. */
const NO_HOLES: number[] = [];

/** A face with its Newell normal, in a height frame for edge matching. */
function newFace(xyz: Float64Array, at: number, count: number, holes: number[], frame: number, yOff: number): Face {
  return { xyz, at, count, holes, frame, yOff, n: newellNormal(xyz, at, holes.length ? holes[0] : count), inner: [] };
}

/** Polygon `polygon` of a 3D feature as a face, its vertices converted to tile-local metres. */
function sourceFace(src: SourceFeature, polygon: number, pow: number, tileScale: number): Face {
  const { xyz, ringEnd, polygonEnd } = src.geometry;
  const outer = polygonEnd[polygon - 1];
  const first = ringEnd[outer - 1];
  const count = ringEnd[polygonEnd[polygon] - 1] - first;
  let holes = NO_HOLES;
  if (polygonEnd[polygon] > outer + 1) {
    holes = [];
    for (let r = outer; r < polygonEnd[polygon] - 1; r++) holes.push(ringEnd[r] - first);
  }
  toTileLocal(xyz, first * 3, count, src.baseZ, src.extent, pow, tileScale);
  return newFace(xyz, first * 3, count, holes, 0, src.baseZ / 10);
}

/** The vertices of a face's outer ring. */
function outerCount(f: Face): number {
  return f.holes.length ? f.holes[0] : f.count;
}

/**
 * Emit one building: fake-AO distances, triangles, and its wall feet for the
 * skirt. Members become consecutive feature spans.
 */
function emitBuilding(out: MeshBuilder, building: Building) {
  const { members, lng, lat, anchor, base } = building;
  const faces = members.flatMap((m) => m.faces);
  for (const member of members) {
    const featureStart = out.vertexCount;
    for (const face of member.faces) emitFace(out, face);
    const span: FeatureSpan = {
      lng,
      lat,
      start: featureStart,
      count: out.vertexCount - featureStart,
      properties: member.properties,
    };
    if (anchor !== undefined) {
      span.anchor = anchor;
      span.base = base;
    }
    if (building.rigid) span.rigid = true;
    out.features.push(span);
  }

  for (const f of faces) collectGroundEdges(out, f);
}

/**
 * How buildings get onto the terrain together. The 3D source gives a building's
 * heights in its own frame and the building's base in it — its lowest point.
 * Buildings step up a slope: most wall feet stand above that base, and at the
 * building's centre the terrain is higher than at its lowest point. Putting the
 * base on the terrain height of the centre lifts such a building (walls hang in
 * the air), and each building by a different amount — neighbours no longer meet
 * as they do in the data: an inner edge misses the roof it belongs to, and where
 * the buildings3d build cut away what a neighbour hides, a slit opens.
 *
 * So: buildings that touch — they share the corner of a wall foot (millimetres)
 * — form a group, and the group goes onto the terrain as a whole, fitted to all
 * its wall feet standing on the ground (Fill3DManager.bake). Collected here per
 * group: those feet, each with the length of wall it stands for. Sets `anchor`
 * and `base` of the grouped buildings.
 *
 * Only buildings with a base height other than 0: bodies generated from OSM (and
 * extrusions) carry heights above their own ground, which say nothing about how
 * they stand to a neighbour — they stay anchored one by one at their centre.
 */
function terrainAnchors(buildings: Building[], columns: Columns): TerrainAnchors | undefined {
  interface Foot {
    column: number;
    height: number;
    weight: number;
  }
  const feet: Foot[][] = [];
  const parent: number[] = [];
  const find = (i: number): number => {
    while (parent[i] !== i) i = parent[i] = parent[parent[i]];
    return i;
  };
  const owner = new globalThis.Map<number, number>(); // ground column → a building standing on it
  let any = false;

  buildings.forEach((b, i) => {
    parent.push(i);
    feet.push([]);
    const first = b.members[0]?.faces[0];
    if (!first || first.frame !== 0 || first.yOff === 0 || b.rigid) return;
    const base = first.yOff;
    if (!b.members.every((m) => m.faces.every((f) => f.frame === 0 && f.yOff === base))) return;
    b.base = base;
    any = true;
    for (const m of b.members) {
      for (const f of m.faces) {
        if (Math.abs(f.n[1]) > 0.5) continue; // roofs and floors
        const { xyz, at } = f;
        const end = at + 3 * outerCount(f);
        let lowest = Infinity;
        for (let p = at; p < end; p += 3) if (xyz[p + 1] < lowest) lowest = xyz[p + 1];
        // What the source sank below the base stands on it.
        const above = Math.max(lowest, 0);
        if (above > GROUND_FOOT_MAX_M) continue;
        for (let p = at; p < end; p += 3) {
          const q = p + 3 === end ? at : p + 3;
          if (xyz[p + 1] > lowest + 0.001 || xyz[q + 1] > lowest + 0.001) continue;
          const half = Math.hypot(xyz[q] - xyz[p], xyz[q + 2] - xyz[p + 2]) / 2;
          if (half < 0.05) continue; // vertical edge or degenerate
          for (const v of [p, q]) {
            const column = columns.at(xyz[v], xyz[v + 2]);
            feet[i].push({ column, height: base + above, weight: half });
            const other = owner.get(column);
            if (other === undefined) owner.set(column, i);
            else parent[find(i)] = find(other);
          }
        }
      }
    }
  });
  if (!any) return undefined;

  const groups = new globalThis.Map<number, number>();
  const members: number[][] = [];
  buildings.forEach((b, i) => {
    if (b.base === undefined) return;
    const root = find(i);
    let g = groups.get(root);
    if (g === undefined) {
      groups.set(root, (g = members.length));
      members.push([]);
    }
    members[g].push(i);
    b.anchor = g;
  });
  const count = feet.reduce((n, list) => n + list.length, 0);
  const anchors: TerrainAnchors = {
    start: new Uint32Array(members.length + 1),
    column: new Uint32Array(count),
    height: new Float32Array(count),
    weight: new Float32Array(count),
  };
  let at = 0;
  members.forEach((list, g) => {
    anchors.start[g] = at;
    for (const i of list) {
      for (const foot of feet[i]) {
        anchors.column[at] = foot.column;
        anchors.height[at] = foot.height;
        anchors.weight[at] = foot.weight;
        at++;
      }
    }
  });
  anchors.start[members.length] = at;
  return anchors;
}

/**
 * Find the inner (concave) edges between faces, tile-wide — within a building
 * and between neighbours: inner corners, roof valleys, walls rising from a
 * lower roof. Faces meeting there share an edge:
 *  - with exactly the same end points (millimetres, heights in the face's
 *    frame). The buildings3d build cleans the topology of the CityGML part so
 *    that every contact of two faces is such a pair (vts, branch
 *    prototype/buildings3d-topology);
 *  - or, for vertical edges, on the same column where they overlap:
 *    neighbours of different height meet in a corner whose edges end at
 *    different heights (OSM bodies and extrusions, which the build leaves as
 *    they are).
 * Not found: an edge lying inside another face that has no edge there — a
 * lower roof of an OSM body or extrusion against a wall running on above it.
 * Stretches are recorded on both faces. Returns the number of pairs found.
 */
function findInnerEdges(faces: Face[]): number {
  // The non-vertical edges, by their end points in millimetres (y in the frame),
  // the lexicographically smaller end first, so that both faces of an edge get
  // one key. In columns (six ints per edge) and chained per hash, not as an
  // object per edge in a list per hash: a tile has millions of edges.
  let capacity = 0;
  for (const f of faces) capacity += f.count;
  const ends = new Int32Array(capacity * 6);
  const edgeFace: Face[] = [];
  /** Per edge its two vertices (offsets in its face's `xyz`). */
  const edgeA = new Int32Array(capacity);
  const edgeB = new Int32Array(capacity);
  /** Per hash the first edge with it; `next` chains the others in the order they came, `last[first]` is where the chain ends. */
  const first = new globalThis.Map<number, number>();
  const next = new Int32Array(capacity).fill(-1);
  const last = new Int32Array(capacity);
  let count = 0;
  const columns = new globalThis.Map<number, EdgeRef[]>();
  for (const f of faces) {
    if (f.n[0] === 0 && f.n[1] === 0 && f.n[2] === 0) continue; // degenerate
    const { xyz, holes } = f;
    for (let r = 0, start = f.at; r <= holes.length; r++) {
      const end = f.at + 3 * (r < holes.length ? holes[r] : f.count);
      for (let a = start; a < end; a += 3) {
        const b = a + 3 === end ? start : a + 3;
        const ax = mm(xyz[a]);
        const ay = mm(xyz[a + 1] + f.yOff);
        const az = mm(xyz[a + 2]);
        const bx = mm(xyz[b]);
        const by = mm(xyz[b + 1] + f.yOff);
        const bz = mm(xyz[b + 2]);
        if (ax === bx && az === bz) {
          const e = edgeRef(f, a, b);
          if (e) pushTo(columns, hash4(f.frame, ax, az, 0x7c1), e);
          continue;
        }
        const swap = ax > bx || (ax === bx && (ay > by || (ay === by && az > bz)));
        const o = count * 6;
        ends[o] = swap ? bx : ax;
        ends[o + 1] = swap ? by : ay;
        ends[o + 2] = swap ? bz : az;
        ends[o + 3] = swap ? ax : bx;
        ends[o + 4] = swap ? ay : by;
        ends[o + 5] = swap ? az : bz;
        edgeFace.push(f);
        edgeA[count] = a;
        edgeB[count] = b;
        const h = hash4(hash4(f.frame, ends[o], ends[o + 1], ends[o + 2]), ends[o + 3], ends[o + 4], ends[o + 5]);
        const head = first.get(h);
        if (head === undefined) {
          first.set(h, count);
          last[count] = count;
        } else {
          next[last[head]] = count;
          last[head] = count;
        }
        count++;
      }
      start = end;
    }
  }

  let pairs = 0;
  const record = (e: EdgeRef, lo: number, hi: number, g: EdgeRef) => {
    const k = edgeStrength(e.f.n, g.f.n);
    e.f.inner.push({ key: e, origin: e.origin, dir: e.dir, lo, hi, m: runOff(e), k });
    g.f.inner.push({ key: g, origin: g.origin, dir: g.dir, lo, hi, m: runOff(g), k });
    pairs++;
  };
  /**
   * An edge on its line, once it pairs (null: degenerate): `refAt` is 1 + its
   * place in `refs`, 0 while not computed — few edges pair, and an array with
   * a slot per edge would be a sparse one, slow to write.
   */
  const refs: (EdgeRef | null)[] = [];
  const refAt = new Int32Array(count);
  const refOf = (i: number) => {
    if (refAt[i] === 0) refAt[i] = refs.push(edgeRef(edgeFace[i], edgeA[i], edgeB[i]) ?? null);
    return refs[refAt[i] - 1];
  };

  // Same end points (a hash may be shared by others: compare them).
  for (const head of first.values()) {
    if (next[head] === -1) continue;
    for (let i = head; i !== -1; i = next[i]) {
      const f = edgeFace[i];
      const o = i * 6;
      for (let j = next[i]; j !== -1; j = next[j]) {
        const g = edgeFace[j];
        if (g === f || g.frame !== f.frame) continue;
        const q = j * 6;
        if (ends[q] !== ends[o] || ends[q + 1] !== ends[o + 1] || ends[q + 2] !== ends[o + 2] || ends[q + 3] !== ends[o + 3] || ends[q + 4] !== ends[o + 4] || ends[q + 5] !== ends[o + 5]) continue;
        const e = refOf(i);
        const h = refOf(j);
        if (!e || !h || !isInnerEdge(e, h)) continue;
        record(e, e.t0, e.t1, h);
      }
    }
  }

  // Columns: sweep up each one, comparing overlapping edges only.
  for (const list of columns.values()) {
    if (list.length < 2) continue;
    list.sort((p, q) => p.t0 - q.t0);
    for (let i = 0; i < list.length; i++) {
      const e = list[i];
      for (let j = i + 1; j < list.length && list[j].t0 < e.t1 - MIN_SHARED_M; j++) {
        const g = list[j];
        if (g.f === e.f || g.f.frame !== e.f.frame || mm(g.f.xyz[g.a]) !== mm(e.f.xyz[e.a]) || mm(g.f.xyz[g.a + 2]) !== mm(e.f.xyz[e.a + 2])) continue;
        const lo = Math.max(e.t0, g.t0);
        const hi = Math.min(e.t1, g.t1);
        if (hi - lo < MIN_SHARED_M || !isInnerEdge(e, g)) continue;
        record(e, lo, hi, g);
      }
    }
  }
  return pairs;
}

/**
 * How much of the sky one face hides from the other at their inner edge,
 * relative to a right angle: in the edge's cross-section, the other face covers
 * (1 + cos α) / 2 of the (cosine-weighted) sky for an opening angle α. So
 * 1 + cos α — 0 for a flat edge, 0.5 at 120°, 1 at 90°, 1.7 at 45°. Both
 * normals point inward, so cos α is minus their dot product.
 */
function edgeStrength(a: Vec3, b: Vec3): number {
  return Math.min(2, Math.max(0, 1 - (a[0] * b[0] + a[1] * b[1] + a[2] * b[2])));
}

/** Metres to whole millimetres. */
function mm(v: number): number {
  return Math.round(v * 1000);
}

/** Append to the list under key k. */
function pushTo<K, V>(map: globalThis.Map<K, V[]>, k: K, v: V) {
  const list = map.get(k);
  if (list) list.push(v);
  else map.set(k, [v]);
}

/** An edge on its line, or undefined if degenerate. */
function edgeRef(f: Face, a: number, b: number): EdgeRef | undefined {
  // Work in the face's height frame, from the lexicographically smaller end, so
  // that two faces computing the same edge get bit-identical lines.
  const xyz = f.xyz;
  const ax = xyz[a];
  const ay = xyz[a + 1] + f.yOff;
  const az = xyz[a + 2];
  const bx = xyz[b];
  const by = xyz[b + 1] + f.yOff;
  const bz = xyz[b + 2];
  const swap = ax > bx || (ax === bx && (ay > by || (ay === by && az > bz)));
  const x0 = swap ? bx : ax;
  const y0 = swap ? by : ay;
  const z0 = swap ? bz : az;
  let dx = (swap ? ax : bx) - x0;
  let dy = (swap ? ay : by) - y0;
  let dz = (swap ? az : bz) - z0;
  const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (len < MIN_SHARED_M) return undefined;
  dx /= len;
  dy /= len;
  dz /= len;
  const d: Vec3 = [dx, dy, dz];
  const t0 = x0 * dx + y0 * dy + z0 * dz;
  const origin: Vec3 = [x0 - t0 * dx, y0 - t0 * dy, z0 - t0 * dz];
  return { f, a, b, dir: d, t0, t1: t0 + len, origin };
}

/**
 * A 29-bit hash of a few integers: a small integer (V8 keeps those below 2^30
 * unboxed, also with pointer compression), the fastest kind of Map key.
 * Collisions only add candidates; every user compares them.
 */
function hashInts(q: number[]): number {
  let h = 0x811c9dc5;
  for (const v of q) h = Math.imul(h ^ v, 0x01000193) ^ (h >>> 15);
  return h & 0x1fffffff;
}

/** hashInts of four integers, without an array. */
function hash4(a: number, b: number, c: number, d: number): number {
  let h = 0x811c9dc5;
  h = Math.imul(h ^ a, 0x01000193) ^ (h >>> 15);
  h = Math.imul(h ^ b, 0x01000193) ^ (h >>> 15);
  h = Math.imul(h ^ c, 0x01000193) ^ (h >>> 15);
  h = Math.imul(h ^ d, 0x01000193) ^ (h >>> 15);
  return h & 0x1fffffff;
}

/**
 * Do two faces meeting along a shared stretch form an inner (concave) edge?
 * Each face runs off from the edge in a direction its winding fixes (its
 * Newell normal × the edge as the face traverses it); the edge is inner when
 * each face runs off to the front of the other. Exact, unlike comparing
 * centroids, which fails for large or non-convex faces.
 */
function isInnerEdge(e: EdgeRef, g: EdgeRef): boolean {
  const nf = e.f.n;
  const ng = g.f.n;
  if (nf[0] * ng[0] + nf[1] * ng[1] + nf[2] * ng[2] > COPLANAR_COS) return false;
  const me = runOff(e);
  const mg = runOff(g);
  // Outward normals are the negated Newell normals.
  const gFront = -(mg[0] * nf[0] + mg[1] * nf[1] + mg[2] * nf[2]);
  const eFront = -(me[0] * ng[0] + me[1] * ng[1] + me[2] * ng[2]);
  return gFront > CONCAVE_EPSILON && eFront > CONCAVE_EPSILON;
}

/** Unit direction, within the face's plane, in which the face runs off from edge e. */
function runOff(e: EdgeRef): Vec3 {
  const { n, xyz } = e.f;
  const d: Vec3 = [xyz[e.b] - xyz[e.a], xyz[e.b + 1] - xyz[e.a + 1], xyz[e.b + 2] - xyz[e.a + 2]];
  const m = cross(n, d);
  const l = Math.hypot(m[0], m[1], m[2]) || 1;
  return [m[0] / l, m[1] / l, m[2] / l];
}

/**
 * The face's inner stretches, merged per edge where they touch, longest
 * first — e.g. a wall split into storeys meets its neighbour as one stretch.
 */
function mergedStretches(face: Face): InnerStretch[] {
  if (face.inner.length < 2) return face.inner; // the common case: nothing to merge
  const byEdge = new globalThis.Map<object, InnerStretch[]>();
  for (const s of face.inner) {
    const list = byEdge.get(s.key);
    if (list) list.push(s);
    else byEdge.set(s.key, [s]);
  }
  const merged: InnerStretch[] = [];
  for (const list of byEdge.values()) {
    list.sort((p, q) => p.lo - q.lo);
    let cur = { ...list[0] };
    for (let i = 1; i < list.length; i++) {
      if (list[i].lo <= cur.hi + MERGE_GAP_M) {
        cur.hi = Math.max(cur.hi, list[i].hi);
        cur.k = Math.max(cur.k, list[i].k); // met by faces at different angles: the stronger
      } else {
        merged.push(cur);
        cur = { ...list[i] };
      }
    }
    merged.push(cur);
  }
  return merged.sort((p, q) => q.hi - q.lo - (p.hi - p.lo));
}

/**
 * Register a face's inner stretches in the AO table, in an orthonormal basis of
 * its plane around the centre of its box. Returns its table index and how to
 * map a vertex into that plane (metres).
 */
function registerAoFace(ao: AoAttributes, face: Face, stretches: InnerStretch[]): number {
  const { n, xyz, at } = face;
  const p0x = xyz[at];
  const p0y = xyz[at + 1];
  const p0z = xyz[at + 2];
  // e1: the outer ring's first edge, made orthogonal to n; e2 = n × e1.
  let e1: Vec3 = [1, 0, 0];
  for (let o = at + 3, end = at + 3 * outerCount(face); o < end; o += 3) {
    const d: Vec3 = [xyz[o] - p0x, xyz[o + 1] - p0y, xyz[o + 2] - p0z];
    const k = d[0] * n[0] + d[1] * n[1] + d[2] * n[2];
    const t: Vec3 = [d[0] - k * n[0], d[1] - k * n[1], d[2] - k * n[2]];
    const l = Math.hypot(t[0], t[1], t[2]);
    if (l > 1e-6) {
      e1 = [t[0] / l, t[1] / l, t[2] / l];
      break;
    }
  }
  const e2 = cross(n, e1);
  const [e1x, e1y, e1z] = e1;
  const [e2x, e2y, e2z] = e2;
  let a0 = Infinity;
  let b0 = Infinity;
  let a1 = -Infinity;
  let b1 = -Infinity;
  for (let o = at, end = at + 3 * face.count; o < end; o += 3) {
    const d0 = xyz[o] - p0x;
    const d1 = xyz[o + 1] - p0y;
    const d2 = xyz[o + 2] - p0z;
    const a = d0 * e1x + d1 * e1y + d2 * e1z;
    const b = d0 * e2x + d1 * e2y + d2 * e2z;
    if (a < a0) a0 = a;
    if (a > a1) a1 = a;
    if (b < b0) b0 = b;
    if (b > b1) b1 = b;
  }
  const ca = (a0 + a1) / 2;
  const cb = (b0 + b1) / 2;
  const edges: PlaneEdge[] = stretches.map((st) => {
    // The stretch start in the face's base-relative coordinates (the line lives in its height frame).
    const t = st.lo;
    const d0 = st.origin[0] + st.dir[0] * t - p0x;
    const d1 = st.origin[1] + st.dir[1] * t - face.yOff - p0y;
    const d2 = st.origin[2] + st.dir[2] * t - p0z;
    const a = d0 * e1x + d1 * e1y + d2 * e1z;
    const b = d0 * e2x + d1 * e2y + d2 * e2z;
    let ua = st.dir[0] * e1[0] + st.dir[1] * e1[1] + st.dir[2] * e1[2];
    let ub = st.dir[0] * e2[0] + st.dir[1] * e2[1] + st.dir[2] * e2[2];
    const l = Math.hypot(ua, ub) || 1;
    ua /= l;
    ub /= l;
    // Which side of the edge's direction the face lies on (where m points).
    const ma = st.m[0] * e1[0] + st.m[1] * e1[1] + st.m[2] * e1[2];
    const mb = st.m[0] * e2[0] + st.m[1] * e2[1] + st.m[2] * e2[2];
    return { a: a - ca, b: b - cb, ua, ub, len: st.hi - st.lo, side: -ub * ma + ua * mb >= 0 ? 1 : -1, k: st.k };
  });
  // How emitFaceAo maps a vertex into the plane (see PLANE).
  PLANE[0] = p0x;
  PLANE[1] = p0y;
  PLANE[2] = p0z;
  PLANE[3] = e1x;
  PLANE[4] = e1y;
  PLANE[5] = e1z;
  PLANE[6] = e2x;
  PLANE[7] = e2y;
  PLANE[8] = e2z;
  PLANE[9] = ca;
  PLANE[10] = cb;
  return ao.tables.add(edges, [a0 - ca, b0 - cb, a1 - ca, b1 - cb], ao.radius);
}

/**
 * The plane of the face registerAoFace was called for last: a point of it, its
 * two axes, the centre of the face's box in it. A vertex p lies at
 * ((p - point) · axis1 - centre1, (p - point) · axis2 - centre2). Kept here
 * instead of handed back, because that would be an object per face.
 */
const PLANE = new Float64Array(11);

/** Scratch of emitFace: a face's outline in 2D for earcut. */
const FLAT_2D: number[] = [];
const ONE_TRIANGLE = [0, 1, 2];

/** A plane coordinate (m) as an aoUV integer. */
function uvUnits(v: number): number {
  return Math.max(-32767, Math.min(32767, Math.round(v / AO_UV_UNIT_M)));
}

/**
 * A face's AO attributes, for its vertices `first` and following: its entry in
 * the AO tables and where each vertex lies in its plane. A face without inner
 * edges leaves them as they are.
 */
function emitFaceAo(ao: AoAttributes, face: Face, first: number) {
  const stretches = mergedStretches(face);
  if (!stretches.length) return;
  const aoIndex = registerAoFace(ao, face, stretches);
  if (aoIndex === AO_NO_FACE) return;
  const [px, py, pz, e1x, e1y, e1z, e2x, e2y, e2z, ca, cb] = PLANE;
  const xyz = face.xyz;
  for (let o = face.at, end = o + 3 * face.count, v = first; o < end; o += 3, v++) {
    const d0 = xyz[o] - px;
    const d1 = xyz[o + 1] - py;
    const d2 = xyz[o + 2] - pz;
    ao.face[v] = aoIndex;
    ao.uv[2 * v] = uvUnits(d0 * e1x + d1 * e1y + d2 * e1z - ca);
    ao.uv[2 * v + 1] = uvUnits(d0 * e2x + d1 * e2y + d2 * e2z - cb);
  }
}

/** Push a face's vertices (with their AO attributes) and triangles. */
function emitFace(out: MeshBuilder, face: Face) {
  const { n, xyz } = face;
  const end = face.at + 3 * face.count;
  const { position, normal, column, columns } = out;
  const first = out.vertexCount;
  out.vertexCount += face.count;
  if (out.ao) emitFaceAo(out.ao, face, first);

  // The plane ⟂ the normal, for earcut: an axis in it (from an arbitrary
  // direction, made orthogonal to the normal) and the one across.
  let x0 = 1.1;
  let x1 = 1.1;
  let x2 = 1.1;
  const d = x0 * n[0] + x1 * n[1] + x2 * n[2];
  x0 -= n[0] * d;
  x1 -= n[1] * d;
  x2 -= n[2] * d;
  const xl = Math.hypot(x0, x1, x2) || 1;
  x0 /= xl;
  x1 /= xl;
  x2 /= xl;
  const y0 = n[1] * x2 - n[2] * x1;
  const y1 = n[2] * x0 - n[0] * x2;
  const y2 = n[0] * x1 - n[1] * x0;

  // The normal as the result has it: scaled to ±127.
  const n0 = Math.round(n[0] * 127);
  const n1 = Math.round(n[1] * 127);
  const n2 = Math.round(n[2] * 127);
  const flat2D = FLAT_2D;
  flat2D.length = 0;
  for (let o = face.at, v = first; o < end; o += 3, v++) {
    const x = xyz[o];
    const y = xyz[o + 1];
    const z = xyz[o + 2];
    position[3 * v] = x;
    position[3 * v + 1] = y;
    position[3 * v + 2] = z;
    normal[3 * v] = n0;
    normal[3 * v + 1] = n1;
    normal[3 * v + 2] = n2;
    column[v] = columns.at(x, z);
    if (out.ao) out.aoh[v] = dm(y);
    flat2D.push(x * x0 + y * x1 + z * x2, x * y0 + y * y1 + z * y2);
  }
  out.addTriangles(face.count === 3 ? ONE_TRIANGLE : earcut(flat2D, face.holes, 2), first);
}

/** Metres → decimetres, clamped into Int16. */
function dm(m: number): number {
  return Math.max(-32766, Math.min(32766, Math.round(m * 10)));
}

/** A wall's edges at the building's foot, for the ground skirt. */
function collectGroundEdges(out: MeshBuilder, f: Face) {
  const n = f.n;
  if (Math.abs(n[1]) > 0.5) return; // roofs and floors
  const hl = Math.hypot(n[0], n[2]);
  if (hl < 1e-6) return;
  const nx = -n[0] / hl; // outward, horizontal
  const nz = -n[2] / hl;
  const { xyz, at } = f;
  for (let a = at, end = at + 3 * outerCount(f); a < end; a += 3) {
    const b = a + 3 === end ? at : a + 3;
    if (xyz[a + 1] > BASE_EPSILON_M || xyz[b + 1] > BASE_EPSILON_M) continue;
    if (Math.hypot(xyz[b] - xyz[a], xyz[b + 2] - xyz[a + 2]) < 0.1) continue; // vertical edge or degenerate
    out.ground.push({ ax: xyz[a], az: xyz[a + 2], bx: xyz[b], bz: xyz[b + 2], nx, nz });
  }
}

/**
 * The ground skirt: along the wall feet a strip `radius` wide on the outward
 * side. Feet that meet at a corner form one strip, mitred there (no overlap,
 * no gap) and sharing the corner's vertices; a free end gets a rounded cap.
 * Where strips still overlap, the renderer keeps the strongest shade (they
 * don't add up). Cut into short pieces so it can follow the terrain.
 */
function buildSkirt(ground: GroundEdge[], radius: number, columns: Columns): SkirtResult {
  // Orient each foot so that its outward normal is on its right-hand side…
  // (east, north) frame: right of direction d is (d.z, -d.x).
  for (const g of ground) {
    const dx = g.bx - g.ax;
    const dz = g.bz - g.az;
    if (dz * g.nx - dx * g.nz < 0) {
      [g.ax, g.bx] = [g.bx, g.ax];
      [g.az, g.bz] = [g.bz, g.az];
    }
  }
  // …so a foot's end meets its successor's start at a corner.
  const key = (x: number, z: number) => `${Math.round(x * 100)},${Math.round(z * 100)}`;
  const feet = dropPartyWalls(ground);

  const starts = new globalThis.Map<string, GroundEdge[]>();
  const ends = new globalThis.Map<string, GroundEdge[]>();
  for (const g of feet) {
    const ks = key(g.ax, g.az);
    const ke = key(g.bx, g.bz);
    (starts.get(ks) ?? starts.set(ks, []).get(ks)!).push(g);
    (ends.get(ke) ?? ends.set(ke, []).get(ke)!).push(g);
  }
  // A corner joins two feet only if exactly one ends and one starts there,
  // and the strip doesn't double back on itself.
  const joined = (p: GroundEdge, n: GroundEdge) => p.nx * n.nx + p.nz * n.nz > -0.9;
  const nextOf = (g: GroundEdge) => {
    const k = key(g.bx, g.bz);
    const s = starts.get(k);
    return s?.length === 1 && ends.get(k)?.length === 1 && joined(g, s[0]) ? s[0] : undefined;
  };
  const prevOf = (g: GroundEdge) => {
    const k = key(g.ax, g.az);
    const e = ends.get(k);
    return e?.length === 1 && starts.get(k)?.length === 1 && joined(e[0], g) ? e[0] : undefined;
  };

  /** Outward offset of the strip's outer edge at a corner between p and n. */
  const mitre = (p: GroundEdge | undefined, n: GroundEdge): [number, number] => {
    if (!p) return [n.nx * radius, n.nz * radius];
    let mx = p.nx + n.nx;
    let mz = p.nz + n.nz;
    const ml = Math.hypot(mx, mz);
    mx /= ml;
    mz /= ml;
    const len = Math.min(radius / Math.max(mx * n.nx + mz * n.nz, 1e-6), radius * SKIRT_MITRE_LIMIT);
    return [mx * len, mz * len];
  };

  const position: number[] = [];
  const coord: number[] = [];
  const index: number[] = [];
  const column: number[] = [];
  /** u: beyond a free end, v: out from the foot — in radii, as bytes. */
  const vertex = (x: number, z: number, u: number, v: number) => {
    position.push(x, 0, z);
    coord.push(u, v);
    // The inner edge lies on the wall feet: it shares their columns.
    column.push(columns.at(x, z));
    return position.length / 3 - 1;
  };
  /** Two triangles between the (inner, outer) vertex pairs a and b. */
  const quad = (a: [number, number], b: [number, number]) => {
    index.push(a[0], b[0], b[1], a[0], b[1], a[1]);
  };

  const visited = new Set<GroundEdge>();
  for (const g0 of feet) {
    if (visited.has(g0)) continue;
    // Walk back to the start of g0's strip (or once round a closed one).
    let first = g0;
    for (let p = prevOf(first), i = 0; p && p !== g0 && !visited.has(p) && i < feet.length; p = prevOf(p), i++) first = p;
    const walked: GroundEdge[] = [];
    let closed = false;
    for (let g: GroundEdge | undefined = first; g; ) {
      walked.push(g);
      visited.add(g);
      const n = nextOf(g);
      if (n === first) closed = true;
      g = n && !visited.has(n) ? n : undefined;
    }
    const chain = mergeStraight(walked, closed);

    // One (inner, outer) vertex pair per station along the strip.
    let firstPair: [number, number] | undefined;
    let prevPair: [number, number] | undefined;
    for (let i = 0; i < chain.length; i++) {
      const g = chain[i];
      const before = i > 0 ? chain[i - 1] : closed ? chain[chain.length - 1] : undefined;
      const after = i < chain.length - 1 ? chain[i + 1] : closed ? chain[0] : undefined;
      const [oax, oaz] = mitre(before, g);
      const [obx, obz] = after ? mitre(g, after) : [g.nx * radius, g.nz * radius];
      const steps = Math.max(1, Math.ceil(Math.hypot(g.bx - g.ax, g.bz - g.az) / SKIRT_STEP_M));
      for (let s = i === 0 ? 0 : 1; s <= steps; s++) {
        let pair: [number, number];
        if (closed && i === chain.length - 1 && s === steps) {
          pair = firstPair!; // back at the start
        } else {
          const t = s / steps;
          const ix = g.ax + (g.bx - g.ax) * t;
          const iz = g.az + (g.bz - g.az) * t;
          pair = [vertex(ix, iz, 0, 0), vertex(ix + oax + (obx - oax) * t, iz + oaz + (obz - oaz) * t, 0, 255)];
        }
        if (prevPair) quad(prevPair, pair);
        prevPair = pair;
        firstPair ??= pair;
      }
    }

    // A free end gets a rounded cap one radius long: it closes the gap to a
    // neighbour's strip; where they overlap, the stronger shade wins.
    if (!closed) {
      const cap = (g: GroundEdge, atStart: boolean, pair: [number, number]) => {
        const L = Math.hypot(g.bx - g.ax, g.bz - g.az);
        const sign = atStart ? -1 : 1;
        const x = (atStart ? g.ax : g.bx) + ((sign * (g.bx - g.ax)) / L) * radius;
        const z = (atStart ? g.az : g.bz) + ((sign * (g.bz - g.az)) / L) * radius;
        quad(pair, [vertex(x, z, 255, 0), vertex(x + g.nx * radius, z + g.nz * radius, 255, 255)]);
      };
      cap(chain[0], true, firstPair!);
      cap(chain[chain.length - 1], false, prevPair!);
    }
  }
  const vertexCount = position.length / 3;
  return {
    position: new Float32Array(position),
    coord: new Uint8Array(coord),
    index: vertexCount <= 65536 ? new Uint16Array(index) : new Uint32Array(index),
    column: new Uint32Array(column),
  };
}

/**
 * Wall feet facing each other on one line — party walls between neighbours,
 * walls between the parts of a building — lie under the other building, and
 * their skirts would show beyond the facade as dark spikes. Cut out every
 * stretch a foot shares with an opposite-facing one, also where the two are
 * split differently (exact twins alone missed those).
 */
function dropPartyWalls(ground: GroundEdge[]): GroundEdge[] {
  interface Line { g: GroundEdge; dx: number; dz: number; t0: number; t1: number; q: number[] }
  const groups = new globalThis.Map<number, Line[]>();
  const lines: Line[] = [];
  for (const g of ground) {
    let dx = g.bx - g.ax;
    let dz = g.bz - g.az;
    const len = Math.hypot(dx, dz);
    dx /= len;
    dz /= len;
    if (dx < 0 || (dx === 0 && dz < 0)) {
      dx = -dx;
      dz = -dz;
    }
    const ta = g.ax * dx + g.az * dz;
    const tb = g.bx * dx + g.bz * dz;
    // Direction to ~0.3°, offset of the line from the origin to 5 cm.
    const q = [Math.round(dx * 200), Math.round(dz * 200), Math.round((g.ax * dz - g.az * dx) * 20)];
    const line = { g, dx, dz, t0: Math.min(ta, tb), t1: Math.max(ta, tb), q };
    lines.push(line);
    const k = hashInts(q);
    const list = groups.get(k);
    if (list) list.push(line);
    else groups.set(k, [line]);
  }

  const out: GroundEdge[] = [];
  for (const l of lines) {
    // Stretches of l covered by opposite-facing feet on the same line.
    const cut: [number, number][] = [];
    for (const o of groups.get(hashInts(l.q))!) {
      if (o === l || o.q[0] !== l.q[0] || o.q[1] !== l.q[1] || o.q[2] !== l.q[2]) continue;
      if (o.g.nx * l.g.nx + o.g.nz * l.g.nz > -0.9) continue;
      const lo = Math.max(l.t0, o.t0);
      const hi = Math.min(l.t1, o.t1);
      if (hi - lo > MIN_SHARED_M) cut.push([lo, hi]);
    }
    if (!cut.length) {
      out.push(l.g);
      continue;
    }
    // What remains of [t0, t1], as feet in the original direction.
    cut.sort((p, q) => p[0] - q[0]);
    const forward = (l.g.bx - l.g.ax) * l.dx + (l.g.bz - l.g.az) * l.dz > 0;
    const ta = l.g.ax * l.dx + l.g.az * l.dz;
    const at = (t: number): [number, number] => [l.g.ax + (t - ta) * l.dx, l.g.az + (t - ta) * l.dz];
    let from = l.t0;
    for (const [lo, hi] of [...cut, [l.t1, l.t1] as [number, number]]) {
      if (lo - from > 0.1) {
        const [px, pz] = at(from);
        const [qx, qz] = at(lo);
        out.push(forward
          ? { ...l.g, ax: px, az: pz, bx: qx, bz: qz }
          : { ...l.g, ax: qx, az: qz, bx: px, bz: pz });
      }
      from = Math.max(from, hi);
    }
  }
  return out;
}

/**
 * Merge consecutive feet of a strip that run on straight (a facade split into
 * several walls, or row houses after their party walls were dropped): one
 * station per real corner instead of one per wall.
 */
function mergeStraight(chain: GroundEdge[], closed: boolean): GroundEdge[] {
  const straight = (p: GroundEdge, n: GroundEdge) => p.nx * n.nx + p.nz * n.nz > SKIRT_STRAIGHT_COS;
  const out: GroundEdge[] = [];
  for (const g of chain) {
    const last = out[out.length - 1];
    if (last && straight(last, g)) {
      last.bx = g.bx;
      last.bz = g.bz;
    } else {
      out.push({ ...g });
    }
  }
  if (closed && out.length > 2 && straight(out[out.length - 1], out[0])) {
    const last = out.pop()!;
    out[0].ax = last.ax;
    out[0].az = last.az;
  }
  return out;
}

/** Cached per worker: the extrusion height/base evaluators, keyed by their spec. */
let extrusionEvals: { spec: string; height: Evaluator<number>; base: Evaluator<number> } | undefined;

/**
 * Extrude the 2D footprints of `extrusion.sourceLayer` into prisms: a flat
 * roof and one wall per ring edge. Footprints are clipped to the tile (the
 * source overhangs it by a buffer); cut edges on the tile border get no wall.
 */
function addExtrusions(
  buildings: { members: Member[]; lng: number; lat: number }[],
  tile: VectorTile,
  extrusion: ExtrusionRequest,
  tx: number,
  ty: number,
  tz: number,
  pow: number,
  tileScale: number,
) {
  const layer = tile.layers[extrusion.sourceLayer];
  if (!layer) return;
  const spec = JSON.stringify([extrusion.height, extrusion.base]);
  if (extrusionEvals?.spec !== spec) {
    extrusionEvals = {
      spec,
      height: compileNumber(extrusion.height ?? ['get', 'height'], 'height', 0),
      base: compileNumber(extrusion.base ?? ['coalesce', ['get', 'min_height'], 0], 'base', 0),
    };
  }
  const { height, base } = extrusionEvals;
  const ff = extrusion.filter ? featureFilter(extrusion.filter, 'filter') : null;
  const globals = { zoom: tz } as GlobalProperties;

  for (let i = 0; i < layer.length; i++) {
    const feature = layer.feature(i);
    if (feature.type !== 3) continue;
    if (ff && !ff.filter(globals, feature as unknown as Feature)) continue;
    const props = feature.properties;
    const top = Number(height(tz, props));
    const bottom = Math.max(0, Number(base(tz, props)) || 0);
    if (!(top > bottom)) continue;

    const extent = feature.extent;
    for (const rings of classifyRings(feature.loadRings2D())) {
      const outer = clipRing(rings[0], 0, extent);
      if (outer.length < 3 || signedArea(outer) <= 0) continue; // outside the tile, or only its border
      const holes = rings.slice(1).map((r) => clipRing(r, 0, extent)).filter((r) => r.length >= 3);
      const clipped = [outer, ...holes];
      const faces = extrude(clipped, bottom, top, extent, pow, tileScale);
      if (!faces.walls.length && !faces.roof) continue;

      const c = ringCentroid2D(clipped[0]);
      const [lng, lat] = mercToLngLat((tx + c.x / extent) / pow, (ty + c.y / extent) / pow);
      buildings.push({
        members: [
          { properties: { ...props, type: 'roof' }, faces: [faces.roof] },
          { properties: { ...props, type: 'wall' }, faces: faces.walls },
        ],
        lng,
        lat,
      });
    }
  }
}

/**
 * Group MVT rings into polygons: a ring with positive area (the MVT winding of
 * an exterior ring, y down) starts a polygon, negative ones are its holes.
 */
function classifyRings(rings: { x: number; y: number }[][]): { x: number; y: number }[][][] {
  const polygons: { x: number; y: number }[][][] = [];
  for (let ring of rings) {
    const last = ring[ring.length - 1];
    if (ring.length > 1 && last.x === ring[0].x && last.y === ring[0].y) ring = ring.slice(0, -1);
    const area = signedArea(ring);
    if (area > 0) polygons.push([ring]);
    else if (area < 0 && polygons.length) polygons[polygons.length - 1].push(ring);
  }
  return polygons;
}

/** Shoelace sum in tile coordinates (y down): positive for MVT exterior rings. */
function signedArea(ring: { x: number; y: number }[]): number {
  let a = 0;
  for (let i = 0, n = ring.length; i < n; i++) {
    const p = ring[i];
    const q = ring[i === n - 1 ? 0 : i + 1];
    a += p.x * q.y - q.x * p.y;
  }
  return a;
}

/** Sutherland–Hodgman clip of a ring to the square [min, max]². Keeps winding. */
function clipRing(ring: { x: number; y: number }[], min: number, max: number): { x: number; y: number }[] {
  type Pt = { x: number; y: number };
  const edge = (pts: Pt[], inside: (p: Pt) => boolean, cut: (a: Pt, b: Pt) => Pt): Pt[] => {
    const res: Pt[] = [];
    for (let i = 0, n = pts.length; i < n; i++) {
      const cur = pts[i];
      const prev = pts[(i + n - 1) % n];
      const ci = inside(cur);
      if (ci !== inside(prev)) res.push(cut(prev, cur));
      if (ci) res.push(cur);
    }
    return res;
  };
  const atX = (x: number) => (a: Pt, b: Pt) => ({ x, y: a.y + ((b.y - a.y) * (x - a.x)) / (b.x - a.x) });
  const atY = (y: number) => (a: Pt, b: Pt) => ({ x: a.x + ((b.x - a.x) * (y - a.y)) / (b.y - a.y), y });
  let pts: Pt[] = ring;
  pts = edge(pts, (p) => p.x >= min, atX(min));
  pts = edge(pts, (p) => p.x <= max, atX(max));
  pts = edge(pts, (p) => p.y >= min, atY(min));
  pts = edge(pts, (p) => p.y <= max, atY(max));
  // Drop repeated points.
  return pts.filter((p, i) => {
    const q = pts[(i + pts.length - 1) % pts.length];
    return pts.length < 2 || p.x !== q.x || p.y !== q.y;
  });
}

/** Roof and walls of one clipped footprint, as faces in tile-local metres. */
function extrude(
  rings: { x: number; y: number }[][],
  bottom: number,
  top: number,
  extent: number,
  pow: number,
  tileScale: number,
): { roof: Face; walls: Face[] } {
  const topZ = Math.round(top * 10); // decimetres, like the 3D source
  const bottomZ = bottom > 0 ? Math.round(bottom * 10) : -EXTRUSION_GROUND_DEPTH_M * 10;
  // The roof's vertices, then four per wall.
  let count = 0;
  for (const ring of rings) count += ring.length;
  const xyz = new Float64Array(15 * count);
  const put = (o: number, p: { x: number; y: number }, z: number) => {
    xyz[o] = p.x;
    xyz[o + 1] = p.y;
    xyz[o + 2] = z;
  };

  // Heights above the extrusion's own ground: frame 1, apart from the 3D source's.
  const holes: number[] = [];
  let at = 0;
  rings.forEach((ring, r) => {
    if (r) holes.push(at / 3);
    for (const p of ring) {
      put(at, p, topZ);
      at += 3;
    }
  });
  toTileLocal(xyz, 0, count, 0, extent, pow, tileScale);
  // The roof's Newell normal must point down (into the building), like the source's.
  if (newellNormal(xyz, 0, rings[0].length)[1] > 0) {
    rings.forEach((ring, r) => reverseRing(xyz, 3 * (r ? holes[r - 1] : 0), ring.length));
  }
  const roof = newFace(xyz, 0, count, holes, 1, 0);

  const walls: Face[] = [];
  const onBorder = (a: { x: number; y: number }, b: { x: number; y: number }) =>
    (a.x === b.x && (a.x <= 0 || a.x >= extent)) || (a.y === b.y && (a.y <= 0 || a.y >= extent));
  for (const ring of rings) {
    for (let i = 0, n = ring.length; i < n; i++) {
      const a = ring[i];
      const b = ring[i === n - 1 ? 0 : i + 1];
      if (onBorder(a, b)) continue; // cut by the tile edge: not a real wall
      // In tile coordinates (y down) the building lies right of a ring's
      // direction for exterior AND hole rings (MVT winding), so (dy, -dx) is outward.
      put(at, a, bottomZ);
      put(at + 3, b, bottomZ);
      put(at + 6, b, topZ);
      put(at + 9, a, topZ);
      toTileLocal(xyz, at, 4, 0, extent, pow, tileScale);
      const nn = newellNormal(xyz, at, 4);
      const outE = b.y - a.y; // outward east  (= dy)
      const outN = b.x - a.x; // outward north (= -(-dx)), tile y grows south
      if (-nn[0] * outE - nn[2] * outN < 0) reverseRing(xyz, at, 4);
      walls.push(newFace(xyz, at, 4, NO_HOLES, 1, 0));
      at += 12;
    }
  }
  return { roof, walls };
}

function ringCentroid2D(ring: { x: number; y: number }[]): { x: number; y: number } {
  let x = 0;
  let y = 0;
  for (const p of ring) {
    x += p.x;
    y += p.y;
  }
  return { x: x / ring.length, y: y / ring.length };
}

/**
 * Split the faces of a building id's features into connected parts: faces
 * whose bounding boxes lie within `tolerance` (extent units) of each other,
 * transitively. A sweep over the boxes sorted by x keeps large city-model
 * buildings (thousands of faces) cheap.
 */
/** All faces of a building id's features as one part. */
function wholePart(group: SourceFeature[]): Part {
  const members: Part['members'] = [];
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const feature of group) {
    const { xyz, ringEnd, polygonEnd } = feature.geometry;
    const polygons: number[] = [];
    for (let polygon = 1; polygon < polygonEnd.length; polygon++) {
      polygons.push(polygon);
      const outer = polygonEnd[polygon - 1];
      for (let o = ringEnd[outer - 1] * 3, end = ringEnd[outer] * 3; o < end; o += 3) {
        const x = xyz[o];
        const y = xyz[o + 1];
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
    members.push({ feature, polygons });
  }
  return { members, minX, minY, maxX, maxY };
}

function connectedParts(group: SourceFeature[], tolerance: number): Part[] {
  const faces: { feature: SourceFeature; polygon: number; box: [number, number, number, number] }[] = [];
  for (const feature of group) {
    const { xyz, ringEnd, polygonEnd } = feature.geometry;
    for (let polygon = 1; polygon < polygonEnd.length; polygon++) {
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      const outer = polygonEnd[polygon - 1];
      for (let o = ringEnd[outer - 1] * 3, end = ringEnd[outer] * 3; o < end; o += 3) {
        const x = xyz[o];
        const y = xyz[o + 1];
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
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

/**
 * `count` vertices from offset `at`, in tile units, become true metres (east,
 * up-above-base, north) from the tile NW corner — in place.
 */
function toTileLocal(
  xyz: Float64Array,
  at: number,
  count: number,
  baseZ: number,
  extent: number,
  pow: number,
  tileScale: number,
) {
  for (let o = at, end = at + 3 * count; o < end; o += 3) {
    const mx = (xyz[o] / extent) / pow; // mercator offset east of the tile corner
    const my = (xyz[o + 1] / extent) / pow; // mercator offset south of the tile corner
    xyz[o] = mx / tileScale; // east metres
    xyz[o + 1] = (xyz[o + 2] - baseZ) / 10; // decimeters → metres, base at 0 (terrain added later)
    xyz[o + 2] = -my / tileScale; // north metres (mercator y grows southward)
  }
}

/** Reverse the order of the `count` vertices from offset `at`. */
function reverseRing(xyz: Float64Array, at: number, count: number) {
  for (let i = at, j = at + 3 * (count - 1); i < j; i += 3, j -= 3) {
    for (let k = 0; k < 3; k++) {
      const v = xyz[i + k];
      xyz[i + k] = xyz[j + k];
      xyz[j + k] = v;
    }
  }
}

/** Newell's method — face normal robust to non-planar rings — for the ring of `count` vertices from offset `at`. */
function newellNormal(xyz: Float64Array, at: number, count: number): Vec3 {
  const n: Vec3 = [0, 0, 0];
  for (let o = at, end = at + 3 * count; o < end; o += 3) {
    const q = o + 3 === end ? at : o + 3;
    n[0] += (xyz[o + 1] - xyz[q + 1]) * (xyz[o + 2] + xyz[q + 2]);
    n[1] += (xyz[o + 2] - xyz[q + 2]) * (xyz[o] + xyz[q]);
    n[2] += (xyz[o] - xyz[q]) * (xyz[o + 1] + xyz[q + 1]);
  }
  const l = Math.hypot(n[0], n[1], n[2]);
  if (l === 0) return n;
  n[0] /= l;
  n[1] /= l;
  n[2] /= l;
  return n;
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/** Inverse web-mercator (normalized 0..1) → [lng, lat] degrees. */
function mercToLngLat(mx: number, my: number): [number, number] {
  const lng = mx * 360 - 180;
  const t = 180 - my * 360;
  const lat = (360 / Math.PI) * Math.atan(Math.exp((t * Math.PI) / 180)) - 90;
  return [lng, lat];
}
