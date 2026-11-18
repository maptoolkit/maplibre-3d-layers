import { MercatorCoordinate, LngLat } from 'maplibre-gl';
import type { CoveringTilesOptions, Map } from 'maplibre-gl';
import * as THREE from 'three';
import { createElevationSampler, createMercatorElevationSampler, tileBounds } from './ElevationCache';
import type { GeoBounds } from './ElevationCache';
import type { ExtrusionRequest, TileResult, TileSpec, AoResult, SkirtResult, FeatureTable, AoTables, TerrainAnchors } from './mvt3dGeometry';
import {
  AO_CELL_M,
  AO_NO_FACE,
  AO_TEXTURE_SHIFT,
  AO_TEXTURE_WIDTH,
  AO_UV_UNIT_M,
  SKIRT_LIFT_M,
  boundingSphere,
  flatGroundOfRigid,
  vertexColors,
} from './mvt3dGeometry';
import type { ColorEval, OpacityEval, PaintValue } from './paint';
import type { Evaluator } from './expression';
import { globalStateOf, watchGlobalState } from './globalState';
import type { FilterSpecification } from '@maplibre/maplibre-gl-style-spec';
import { TileHub } from './TileHub';
import type { PoolWorker, TileConsumer } from './TileHub';
import { dataZooms } from './dataZooms';
import type { PreloadCamera } from './Fill3DLayer';

/** What preload sets on a copy of MapLibre's transform (internal API). */
interface Transform {
  setCenter(center: LngLat): void;
  setZoom(zoom: number): void;
  setPitch(pitch: number): void;
  setBearing(bearing: number): void;
  setPadding(padding: { top: number; bottom: number; left: number; right: number }): void;
}

/** Wait this long (ms) at most for the source's layer list before loading tiles of every zoom level. */
const LAYER_ZOOMS_WAIT_MS = 1500;
/** Tiles are looked at for changed terrain (see terrainMoved) for at most this much of a frame (ms). */
const TERRAIN_CHECK_MS = 4;

/** Remember this many tiles that turned out to hold nothing for the layer. */
const EMPTY_KEYS_MAX = 4096;

/** Vertices this close (metres) above a building's base count as its base. */
const BASE_EPSILON_M = 0.05;

/** Re-anchor the scene origin once the map centre drifts this far from it (metres). */
const REBASE_DISTANCE_M = 3000;

/**
 * While the map moves, look for tiles to load (and to drop) at most this often
 * (ms): a long flight or drag gets its buildings on the way, not only once the
 * move has ended.
 */
const MOVE_UPDATE_MS = 250;

/** Strength of the two fake-AO terms at intensity 1: darkening at the wall foot / in an inner corner. */
const AO_GROUND_WEIGHT = 0.6;
const AO_CORNER_WEIGHT = 0.5;

/** A tile's fake AO fades in over this long (ms) once its data has arrived. */
const AO_FADE_MS = 500;

// Replicates MapLibre's fill-extrusion vertex lighting (a flat directional ramp,
// NOT PBR) so the 3D meshes shade like the native fill-extrusion LOD1. Works in
// sRGB and outputs directly (no colour-management conversion), like MapLibre.
// `normal`/`color` are our per-vertex attributes in the (east, up, north) frame;
// u_lightpos is the style light mapped into that same frame (see syncLight()).
const FILL3D_VERTEX = /* glsl */ `
  attribute vec4 color;
  // Fake AO: the height above ground (dm); the face's entry in the AO tables and
  // the position in its plane (units of AO_UV_UNIT_M), see innerEdges().
  attribute float aoh;
  attribute float aoface;
  attribute vec2 aouv;
  uniform vec3 u_lightpos;
  uniform float u_lightintensity;
  uniform vec3 u_lightcolor;
  uniform float u_opacity; // fill-3d-opacity when it's the same for every feature
  varying vec4 v_color;
  varying float v_aoh;
  // Not \`flat\`, although it is the same for a face's vertices: a flat varying
  // takes its value from one vertex of the triangle, GL from the last and
  // Direct3D from the first, and Chrome on Windows (ANGLE) emulates GL's choice
  // at a cost — measured on an Intel Arc: this one varying as flat made the
  // colour pass 1.4 ms slower (4.2 instead of 2.8 ms). Interpolating equal
  // values gives the value as well — except on sliver triangles, where the
  // interpolation itself gets inexact: single pixels near the horizon then read
  // another face's edges (20 of 809,100 pixels in a Vienna view, shade off by
  // 8 % at most; \`centroid\` does not change that).
  varying highp float v_aoface;
  varying highp vec2 v_aouv;
  void main() {
    v_aoh = aoh * 0.1;
    v_aoface = aoface;
    v_aouv = aouv * ${AO_UV_UNIT_M};
    float luminance = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
    vec3 base = color.rgb + 0.03; // MapLibre adds a small ambient
    // Our (east, up, north) frame is left-handed, so Newell normals come out
    // inward-facing — negate to get outward normals (roofs → up → lit like MapLibre).
    vec3 n = normalize(-normal);
    // fill-extrusion's wall normals point INTO the building (perp of the edge in
    // y-down tile space), so its walls are lit from the far side of the light.
    // Mirror the horizontal part to match; roofs (pointing up) are unaffected.
    n.xz = -n.xz;
    float directional = clamp(dot(n, u_lightpos), 0.0, 1.0);
    directional = mix(1.0 - u_lightintensity, max(1.0 - luminance + u_lightintensity, 1.0), directional);
    vec3 lit = clamp(
      base * directional * u_lightcolor,
      mix(vec3(0.0), vec3(0.3), 1.0 - u_lightcolor),
      vec3(1.0)
    );
    v_color = vec4(lit, color.a * u_opacity);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

// Fake ambient occlusion: darken towards the wall foot and towards inner
// (concave) edges. Distances come per vertex; within a planar face they are
// linear, so the interpolated value is exact per fragment.
const AO_FUNCTIONS = /* glsl */ `
  uniform float u_ao_intensity;
  uniform float u_ao_tile; // 0..1: this tile's AO fading in once its AO data has arrived
  uniform float u_ao_radius;
  // 1 at the edge, falling off to 0 at the radius.
  float occlusion(float d) {
    float t = clamp(d / u_ao_radius, 0.0, 1.0);
    return (1.0 - t) * (1.0 - t);
  }
  // An inner edge (distance into the face, along from its start, along to its
  // end): the distance to the shared stretch, rounded beyond its ends. A
  // negative distance lies beyond the edge's line (a non-convex face).
  float edgeOcclusion(vec3 e) {
    if (e.x < 0.0) return 0.0;
    float beyond = max(max(-e.y, -e.z), 0.0);
    return occlusion(length(vec2(e.x, beyond)));
  }
`;

// The inner edges of the fragment's face near it, from the tile's AO tables
// (see AoTables): the face's grid cell lists them; each darkens by its distance
// (rounded beyond its ends), weighted by how much sky its opening angle hides
// (1 at 90°); the strongest wins. Plane positions need highp.
const AO_EDGES = /* glsl */ `
  uniform highp usampler2D u_ao_faces;
  uniform highp usampler2D u_ao_cells;
  uniform highp usampler2D u_ao_lists;
  uniform highp usampler2D u_ao_edges;
  varying highp float v_aoface; // not flat, see the vertex shader
  varying highp vec2 v_aouv;
  ivec2 aoTexel(int i) {
    return ivec2(i & ${AO_TEXTURE_WIDTH - 1}, i >> ${AO_TEXTURE_SHIFT});
  }
  // One edge's darkening at the fragment.
  float aoEdge(int e) {
    uvec4 q = texelFetch(u_ao_edges, aoTexel(e), 0);
    highp vec2 u = unpackSnorm2x16(q.z);
    vec2 lk = unpackHalf2x16(q.w); // signed length, strength
    highp vec2 d = v_aouv - vec2(uintBitsToFloat(q.x), uintBitsToFloat(q.y));
    highp float along = dot(d, u);
    highp float into = dot(d, sign(lk.x) * vec2(-u.y, u.x));
    return lk.y * edgeOcclusion(vec3(into, along, abs(lk.x) - along));
  }
  float innerEdges() {
    int face = int(v_aoface + 0.5);
    if (face >= ${AO_NO_FACE}) return 0.0;
    uvec4 h = texelFetch(u_ao_faces, aoTexel(face), 0);
    int nx = int(h.y & 65535u);
    int ny = int(h.y >> 16u);
    int firstEdge = int(h.w);
    float occ = 0.0;
    if (nx == 0) {
      // Few edges, no grid: check them all.
      for (int i = 0; i < ny; i++) occ = max(occ, aoEdge(firstEdge + i));
      return occ;
    }
    // Grid origin: two Int16 decimetres.
    highp vec2 origin = vec2(float(int(h.x << 16u) >> 16), float(int(h.x) >> 16)) * 0.1;
    ivec2 c = clamp(ivec2(floor((v_aouv - origin) / ${AO_CELL_M.toFixed(1)})), ivec2(0), ivec2(nx - 1, ny - 1));
    uint cell = texelFetch(u_ao_cells, aoTexel(int(h.z) + c.y * nx + c.x), 0).r;
    int first = int(cell >> 8u);
    int count = int(cell & 255u);
    for (int i = 0; i < count; i++) {
      occ = max(occ, aoEdge(firstEdge + int(texelFetch(u_ao_lists, aoTexel(first + i), 0).r)));
    }
    return occ;
  }
`;

const FILL3D_FRAGMENT = /* glsl */ `
  precision mediump float;
  ${AO_FUNCTIONS}
  ${AO_EDGES}
  varying vec4 v_color;
  varying float v_aoh;
  void main() {
    vec4 color = v_color;
    float ao = u_ao_intensity * u_ao_tile;
    if (ao > 0.0) {
      float ground = occlusion(max(v_aoh, 0.0));
      float corner = innerEdges();
      // The stronger term wins: an inner corner at the wall foot is no darker
      // than either alone (no doubling where edges meet).
      color.rgb *= 1.0 - ao * max(${AO_GROUND_WEIGHT.toFixed(2)} * ground, ${AO_CORNER_WEIGHT.toFixed(2)} * corner);
    }
    gl_FragColor = color;
  }
`;

// The ground skirt darkens what lies under it: most at the wall foot, fading out
// at the skirt radius, rounded beyond a strip's free ends. Where strips overlap
// only the strongest darkening counts (see SkirtMode).
const SKIRT_VERTEX = /* glsl */ `
  attribute vec2 coord; // in skirt radii: beyond a free end of the strip, out from the wall foot
  varying vec2 v_coord;
  void main() {
    v_coord = coord;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const SKIRT_FRAGMENT = /* glsl */ `
  precision mediump float;
  uniform float u_ao_intensity;
  uniform float u_ao_tile; // 0..1: this tile's AO fading in
  uniform float u_opacity;
  uniform int u_mode; // 0 multiply, 1 collect the darkest shade in alpha, 2 apply it
  varying vec2 v_coord;
  void main() {
    float t = clamp(length(v_coord), 0.0, 1.0); // distance from the foot, in radii
    float shade = 1.0 - u_ao_intensity * u_ao_tile * ${AO_GROUND_WEIGHT.toFixed(2)} * u_opacity * (1.0 - t) * (1.0 - t);
    gl_FragColor = u_mode == 0 ? vec4(vec3(shade), 1.0) : u_mode == 1 ? vec4(0.0, 0.0, 0.0, shade) : vec4(0.0, 0.0, 0.0, 1.0);
  }
`;

/**
 * How the skirt is blended over the map.
 *  - 'collect' then 'apply' (two passes): the first keeps per pixel the darkest
 *    shade in the framebuffer's alpha (MIN blending, colour untouched), the
 *    second multiplies the colour by it once and puts alpha back to 1 — so
 *    overlapping strips don't add up.
 *  - 'multiply' (one pass): colour × shade; overlaps darken further. The
 *    fallback for a canvas without an alpha channel.
 */
type SkirtMode = 'multiply' | 'collect' | 'apply';

function setSkirtMode(m: THREE.ShaderMaterial, mode: SkirtMode) {
  m.uniforms.u_mode.value = mode === 'multiply' ? 0 : mode === 'collect' ? 1 : 2;
  m.blendEquation = THREE.AddEquation;
  if (mode === 'multiply') {
    m.blendSrc = THREE.ZeroFactor; // colour = dst × shade
    m.blendDst = THREE.SrcColorFactor;
    m.blendEquationAlpha = THREE.AddEquation;
    m.blendSrcAlpha = THREE.ZeroFactor; // alpha kept
    m.blendDstAlpha = THREE.OneFactor;
  } else if (mode === 'collect') {
    m.blendSrc = THREE.ZeroFactor; // colour kept
    m.blendDst = THREE.OneFactor;
    m.blendEquationAlpha = THREE.MinEquation; // alpha = min(alpha, shade)
    m.blendSrcAlpha = THREE.OneFactor;
    m.blendDstAlpha = THREE.OneFactor;
  } else {
    m.blendSrc = THREE.ZeroFactor; // colour = dst × dst alpha (the darkest shade)…
    m.blendDst = THREE.DstAlphaFactor;
    m.blendEquationAlpha = THREE.AddEquation;
    m.blendSrcAlpha = THREE.OneFactor; // …then alpha = 1, so a second strip on
    m.blendDstAlpha = THREE.ZeroFactor; // the same pixel multiplies by 1
  }
}

// Scratch colours reused by syncLight() (called every frame).
const SCRATCH_COLOR = new THREE.Color();
const SCRATCH_SRGB = new THREE.Color();

/** The AO tables of a tile as textures (see AoTables). */
interface AoTextures {
  faces: THREE.DataTexture;
  cells: THREE.DataTexture;
  lists: THREE.DataTexture;
  edges: THREE.DataTexture;
}

/** What a buffer or texture holds (bytes), also once its copy in memory is given up (see uploadOnce). */
const BYTES = new WeakMap<object, number>();

/**
 * A buffer that is uploaded once and never written again: its copy in memory
 * is given up after the upload. three keeps the array of every attribute, as
 * it may have to upload it again — for a view of Vienna that was about 100 MB
 * of normals, colours, AO attributes and indices held twice.
 */
function uploadOnce(attribute: THREE.BufferAttribute): THREE.BufferAttribute {
  BYTES.set(attribute, (attribute.array as ArrayBufferView).byteLength);
  return attribute.onUpload(releaseArray);
}

function releaseArray(this: THREE.BufferAttribute) {
  (this as unknown as { array: null }).array = null;
}

/** A table as an integer texture of AO_TEXTURE_WIDTH-texel rows, read with texelFetch. */
function aoTexture(data: Uint32Array | Uint16Array, channels: 1 | 4): THREE.DataTexture {
  const short = data instanceof Uint16Array;
  const tex = new THREE.DataTexture(
    data,
    AO_TEXTURE_WIDTH,
    data.length / channels / AO_TEXTURE_WIDTH,
    channels === 4 ? THREE.RGBAIntegerFormat : THREE.RedIntegerFormat,
    short ? THREE.UnsignedShortType : THREE.UnsignedIntType,
  );
  tex.internalFormat = channels === 4 ? 'RGBA32UI' : short ? 'R16UI' : 'R32UI';
  tex.minFilter = THREE.NearestFilter;
  tex.magFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

function tablesAsTextures(t: AoTables): AoTextures {
  return { faces: aoTexture(t.faces, 4), cells: aoTexture(t.cells, 1), lists: aoTexture(t.lists, 1), edges: aoTexture(t.edges, 4) };
}

/** A tile's AO tables as textures; uploaded once, like its static buffers (see uploadOnce). */
function aoTextures(t: AoTables): AoTextures {
  const textures = tablesAsTextures(t);
  for (const tex of Object.values(textures) as THREE.DataTexture[]) {
    BYTES.set(tex, (tex.image.data as ArrayBufferView).byteLength);
    tex.onUpdate = () => {
      (tex.image as { data: unknown }).data = null;
    };
  }
  return textures;
}

/** Empty tables: integer samplers need integer textures bound, also for tiles without AO. */
function emptyAoTextures(): AoTextures {
  const row = (channels: number) => AO_TEXTURE_WIDTH * channels;
  return tablesAsTextures({ faces: new Uint32Array(row(4)), cells: new Uint32Array(row(1)), lists: new Uint16Array(row(1)), edges: new Uint32Array(row(4)) });
}

function createFill3DMaterial(transparent: boolean, ao: AoTextures): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      u_lightpos: { value: new THREE.Vector3(0.288, 0.996, 0.498) },
      u_lightintensity: { value: 0.5 },
      u_lightcolor: { value: new THREE.Vector3(1, 1, 1) },
      u_opacity: { value: 1 },
      u_ao_intensity: { value: 0 },
      u_ao_radius: { value: 3 },
      // Swapped per tile (bindAo).
      u_ao_tile: { value: 0 },
      u_ao_faces: { value: ao.faces },
      u_ao_cells: { value: ao.cells },
      u_ao_lists: { value: ao.lists },
      u_ao_edges: { value: ao.edges },
    },
    vertexShader: FILL3D_VERTEX,
    fragmentShader: FILL3D_FRAGMENT,
    // Backface culling (like fill-extrusion): only camera-facing outer faces
    // draw, so semi-transparent buildings don't show their inner/back walls.
    // BackSide is the outer shell here (source winding + scale(1,1,-1) reflection).
    side: THREE.BackSide,
    transparent,
    // A transparent layer draws after a depth pre-pass and blends only what lies
    // at that depth; an opaque one writes its own depth (see setPass).
    depthFunc: THREE.LessEqualDepth,
    depthWrite: !transparent,
  });
}

/** The depth pre-pass of a transparent layer: a program that only places the vertices. */
function createDepthMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader: 'void main() { gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'void main() {}',
    side: THREE.BackSide, // as the colour material
    colorWrite: false,
  });
}

function createSkirtMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      u_ao_intensity: { value: 0 },
      u_ao_tile: { value: 0 }, // per tile, see bindSkirt
      u_opacity: { value: 1 },
      u_mode: { value: 0 },
    },
    vertexShader: SKIRT_VERTEX,
    fragmentShader: SKIRT_FRAGMENT,
    side: THREE.DoubleSide,
    transparent: true,
    depthWrite: false,
    // Darkens the map (and nothing else) under the skirt; see SkirtMode.
    blending: THREE.CustomBlending,
    polygonOffset: true,
    polygonOffsetFactor: -2,
    polygonOffsetUnits: -8,
  });
}

/** Fake ambient occlusion settings (EXPERIMENTAL). */
export interface Fill3DAoConfig {
  /** 0..1 per frame (zoom / global state); 0 switches it off. */
  intensity: Evaluator<number>;
  /** Metres: how far up the wall foot and how far from an inner edge it reaches. */
  radius: number;
  /** Metres: width of the ground skirt; 0 = no skirt. */
  groundRadius: number;
}

/** Source + paint config for the features. */
export interface Fill3DConfig {
  color: ColorEval;
  opacity: OpacityEval;
  /** Whether opacity/color can be < 1 (→ the layer may blend; whether a frame does, see syncOpacity). */
  transparent: boolean;
  /** Id of the MapLibre vector source to read tiles from. */
  sourceId: string;
  /** Vector-tile layer name (the `source-layer`). */
  sourceLayer: string;
  /** Optional MapLibre filter expression, applied per feature in the worker. */
  filter?: FilterSpecification;
  /** Also extrude a layer of 2D footprints from the same tiles (EXPERIMENTAL). */
  extrusion?: ExtrusionRequest;
  ao: Fill3DAoConfig;
  /**
   * Overlapping skirts: only the strongest darkening counts (two passes via the
   * framebuffer's alpha channel). False for a canvas without alpha: they multiply.
   */
  skirtMax: boolean;
  /** At most this many tile workers (they start as tiles queue up). */
  workers: number;
  /** `fill-3d-color` / `fill-3d-opacity` as given to the layer: the worker evaluates them per tile. */
  colorSpec?: PaintValue<string>;
  opacitySpec?: PaintValue<number>;
  /** Keep up to this many bytes of tiles that left the view (0: none). */
  cacheBytes: number;
}

/** Scene anchor in mercator + the metric at its latitude. */
export interface SceneOrigin {
  x: number;
  y: number;
  z: number;
  s: number;
}

interface TileEntry {
  state: 'loading' | 'ready' | 'empty';
  /** Tile NW corner in mercator (for placing the mesh relative to the origin). */
  tileMx: number;
  tileMy: number;
  /** meterInMercatorCoordinateUnits at the tile latitude (set once parsed). */
  tileScale?: number;
  /** Geographic bounds of the tile — used to rebake only tiles a changed DEM
   *  tile actually overlaps. */
  bounds: GeoBounds;
  mesh?: THREE.Mesh;
  geometry?: THREE.BufferGeometry;
  skirtMesh?: THREE.Mesh;
  skirtGeometry?: THREE.BufferGeometry;
  /** The tile's AO tables (inner edges), bound before its mesh draws. */
  ao?: AoTextures;
  /** Whether any of its vertices carries an alpha below 1 (see colorAttribute). */
  translucent?: boolean;
  /** When the tile's AO data arrived (ms), while it
   *  fades in; and how far it has faded in (0..1; undefined: no AO data yet). */
  aoSince?: number;
  aoFade?: number;
  /** The request's number at the {@link TileHub}. */
  seq?: number;
  /** Its AO data comes in a message of its own (`aoLater`) and is not in yet (`aoDue`). */
  aoLater?: boolean;
  aoDue?: boolean;
  /** What the tile holds in memory (bytes), once it is ready. */
  bytes?: number;
  /** The terrain and paint states (see terrainEpoch, colorEpoch) its heights and colours were made for. */
  terrainEpoch?: number;
  colorEpoch?: number;
  /** Whether it was last baked onto the flat map (no terrain). */
  bakedFlat?: boolean;
  /** Per vertex: the worker's y, the height above the feature base (no terrain
   *  yet). x and z never change, so only y is kept for re-baking. */
  baseY?: Float32Array;
  /**
   * Per vertex / skirt vertex its ground column, per column x, z — and what the
   * tile was last baked onto: the ground height per column and the terrain
   * height at the centre per feature, each where the bake used it (else NaN).
   * New terrain data under the tile only means a new bake if it changes one of
   * these (see terrainMoved).
   */
  column?: Uint16Array | Uint32Array;
  skirtColumn?: Uint32Array;
  columnXZ?: Float32Array;
  groundCache?: Float32Array;
  centreCache?: Float32Array;
  /** The features' vertex spans, and — only if the paint has to be evaluated again on a global-state change — their properties. */
  table?: FeatureTable;
  properties?: Record<string, unknown>[];
  /** The wall feet per group of touching buildings (see anchorFits). */
  anchors?: TerrainAnchors;
}

/**
 * Streams `buildings3d` tiles for the current viewport, parses/triangulates them
 * off-thread, and adds one merged mesh per tile to the shared three.js scene.
 *
 * Geometry is stored TILE-LOCALLY (metres from each tile's NW corner) so it is
 * independent of the scene origin. Each mesh is positioned relative to the
 * current {@link origin}; when the map centre drifts too far, the origin is
 * re-anchored and every mesh repositioned — no re-parsing. This keeps float32
 * precision high anywhere on earth (e.g. flying Vienna → Berlin).
 *
 * Vertical placement: the worker emits feature-base-relative heights; this class
 * bakes terrain elevation into the vertex Y — per group of touching buildings
 * fitted to their wall feet, else per feature centre (see bake) — and re-bakes
 * whenever new DEM tiles stream in or the
 * map's terrain is switched. Without terrain the meshes sit on the flat map,
 * nothing below it: the source sinks ground walls 10 m for slopes (see bake).
 *
 * Loading: tiles come through the source's {@link TileHub} (one fetch and one
 * worker pass per tile for all layers of the source), only from the zoom levels
 * that can hold something for this layer (see dataZooms). A tile that leaves
 * the view is kept, hidden, within `cacheBytes` and shown again without
 * loading; {@link preload} loads a view ahead into that store.
 */
export class Fill3DManager implements TileConsumer {
  /** Current scene anchor; read by the layer to build the projection matrix. */
  origin: SceneOrigin;

  private readonly map: Map;
  private readonly scene: THREE.Scene;
  private readonly group: THREE.Group;
  /** Ground skirts; drawn in the colour pass only, after the buildings' depth. */
  private readonly skirtGroup: THREE.Group;
  private readonly skirtMaterial: THREE.ShaderMaterial;
  private readonly terrainSourceId: string;
  private hub?: TileHub;
  private consumerId = 0;
  /** The view's tiles are being loaded: the source and the zoom levels of its layers are known. */
  private started = false;
  /** The tile zoom levels this layer has anything at (undefined: all of the source's). */
  private zooms?: [number, number];
  /** No zoom level has anything for this layer. */
  private nothing = false;
  /** The source is not there: nothing will ever load. */
  private failed = false;
  private disposed = false;
  /** The tiles of the view. */
  private readonly tiles = new globalThis.Map<string, TileEntry>();
  /**
   * Tiles outside the view, hidden: those that left it (oldest first) and
   * those loaded ahead (see preload). Ready ones count against `cacheBytes`.
   */
  private readonly kept = new globalThis.Map<string, TileEntry>();
  private keptBytes = 0;
  private cacheBytes: number;
  /** Loaded ahead for the view to come: not given up for memory until it has been shown or another view is loaded ahead. */
  private pins = new Set<string>();
  /** Tiles that hold nothing for this layer: not loaded again. */
  private readonly emptyKeys = new Set<string>();
  /** Counted up when the terrain / the paint's global state changes: a kept tile is brought up to date when it is shown again. */
  private terrainEpoch = 0;
  private colorEpoch = 0;
  /** The paint reads global state per feature: it is evaluated again when that changes (needs the features' properties). */
  private readonly repaints: boolean;
  private readonly material: THREE.ShaderMaterial;
  /** Whether the current frame blends (set by syncOpacity; starts as the material is created). */
  private blends: boolean;
  private readonly depthMaterial: THREE.ShaderMaterial;
  /** Empty AO tables, bound for tiles without inner edges. */
  private readonly aoEmpty: AoTextures;
  private readonly config: Fill3DConfig;
  private readonly onData: (ev: unknown) => void;
  private readonly onMoveEnd: () => void;
  private readonly onMove: () => void;
  /** When update() last ran (ms), for the throttle while the map moves. */
  private lastUpdate = -Infinity;
  /** The look at the view that is due once a series of `moveend`s has ended. */
  private moveEndTimer?: ReturnType<typeof setTimeout>;
  private readonly onTerrain: () => void;
  private readonly onContextRestored: () => void;
  private onSourceData?: (ev: unknown) => void;
  private readonly unwatchGlobalState: () => void;
  private recolorRafId?: number;

  // Terrain DEM tiles stream in as a burst of `data` events. Instead of a full
  // rebake per event, we accumulate the changed tiles' bounds and flush ONE
  // rebake on the next animation frame, touching only overlapping building tiles.
  private pendingDemRegions: GeoBounds[] = [];
  private rebakeRafId?: number;
  /** Tiles whose terrain may have changed: checked, and baked anew if it has, one tile per frame. */
  private readonly rebakeQueue = new Set<TileEntry>();

  // Read from the MapLibre vector source once its TileJSON has loaded.
  private tileTemplate?: string;
  private minzoom?: number;
  private maxzoom?: number;

  constructor(
    map: Map,
    scene: THREE.Scene,
    sceneOrigin: LngLat,
    terrainSourceId: string,
    config: Fill3DConfig,
  ) {
    this.map = map;
    this.scene = scene;
    this.terrainSourceId = terrainSourceId;
    this.config = config;
    this.cacheBytes = config.cacheBytes;
    this.origin = toOrigin(sceneOrigin);

    this.group = new THREE.Group();
    scene.add(this.group);
    this.skirtGroup = new THREE.Group();
    scene.add(this.skirtGroup);

    // Shades like MapLibre's fill-extrusion (flat directional ramp). Per-vertex
    // sRGB colour (fill-3d-color) + alpha (fill-3d-opacity) via the `color` attr.
    this.aoEmpty = emptyAoTextures();
    this.material = createFill3DMaterial(config.transparent, this.aoEmpty);
    this.blends = config.transparent;
    this.depthMaterial = createDepthMaterial();
    this.material.uniforms.u_ao_radius.value = config.ao.radius;
    this.skirtMaterial = createSkirtMaterial();


    // DEM tiles stream in → re-bake feature heights against the new terrain.
    this.onData = (ev) => {
      const e = ev as {
        sourceId?: string;
        tile?: { tileID?: { canonical?: { x: number; y: number; z: number } } };
      };
      if (e.sourceId === this.terrainSourceId && e.tile) {
        const c = e.tile.tileID?.canonical;
        if (!c) return;
        this.pendingDemRegions.push(tileBounds(c.x, c.y, c.z));
        this.scheduleRebake();
      }
    };
    this.onMoveEnd = () => {
      // A camera that is set on every frame ends a move on every frame: such a
      // series is looked at as often as a move in progress (see onMove), and
      // once more when it has ended.
      const wait = MOVE_UPDATE_MS - (performance.now() - this.lastUpdate);
      if (wait > 0) {
        this.moveEndTimer ??= setTimeout(() => {
          this.moveEndTimer = undefined;
          if (this.disposed) return;
          this.maybeRebase();
          this.update();
        }, wait);
        return;
      }
      this.maybeRebase();
      this.update();
    };
    this.onMove = () => {
      if (performance.now() - this.lastUpdate < MOVE_UPDATE_MS) return;
      this.maybeRebase();
      this.update();
    };
    // Terrain switched on, off or re-set: the heights are stale, and without
    // terrain the sunk wall bases must come up to the map (see bake).
    this.onTerrain = () => {
      this.terrainEpoch++;
      this.rebakeAll();
    };
    map.on('data', this.onData);
    map.on('moveend', this.onMoveEnd);
    map.on('move', this.onMove);
    map.on('terrain', this.onTerrain);
    // A restored WebGL context has none of the buffers, and those given up after
    // their upload (see uploadOnce) cannot be uploaded again: the tiles are loaded anew.
    this.onContextRestored = () => this.reload();
    map.on('webglcontextrestored', this.onContextRestored);

    // `["global-state", …]`: a per-feature value is baked into the vertices, so
    // re-bake it; a feature-constant opacity is a uniform read every frame.
    const { color, opacity } = config;
    const baked = [...color.globalStateRefs, ...(opacity.featureConstant ? [] : opacity.globalStateRefs)];
    this.repaints = baked.length > 0;
    this.unwatchGlobalState = watchGlobalState(map, [...baked, ...opacity.globalStateRefs], () => {
      if (baked.length) this.scheduleRecolor();
      else map.triggerRepaint();
    });

    this.initSource();
  }

  /**
   * Read the tile URL template + zoom range from the (already-added) MapLibre
   * vector source `config.sourceId`, waiting for its TileJSON to load. We fetch
   * the ORIGINAL tiles ourselves rather than reading MapLibre's tile bytes,
   * because above `maxzoom` MapLibre overzooms tiles — re-slicing/re-encoding
   * only the style-consumed layers and dropping our `buildings3d` layer entirely.
   */
  private initSource() {
    const id = this.config.sourceId;

    const tryRead = (): boolean => {
      const src = this.map.getSource(id) as unknown as
        | { tiles?: string[]; minzoom?: number; maxzoom?: number }
        | undefined;
      if (!src) return false;
      const template = src.tiles?.[0];
      if (!template || src.minzoom == null || src.maxzoom == null) return false;
      if (!this.map.isSourceLoaded(id)) return false;
      this.tileTemplate = template;
      this.minzoom = src.minzoom;
      this.maxzoom = src.maxzoom;
      return true;
    };

    if (!this.map.getSource(id)) {
      console.error(`Fill3DLayer: source '${id}' not found — add it before the layer`);
      this.failed = true;
      return;
    }
    if (tryRead()) {
      this.start();
      return;
    }
    this.onSourceData = (ev) => {
      if ((ev as { sourceId?: string }).sourceId !== id) return;
      if (tryRead()) {
        this.map.off('sourcedata', this.onSourceData!);
        this.onSourceData = undefined;
        this.start();
      }
    };
    this.map.on('sourcedata', this.onSourceData);
  }

  /**
   * The source is known: join its {@link TileHub} and find the zoom levels the
   * layer has anything at (the source's layer list is waited for, briefly),
   * then load the view.
   */
  private start() {
    this.hub = TileHub.for(this.map, this.config.sourceId);
    this.consumerId = this.hub.attach(this, this.config.workers);
    const go = () => {
      if (this.started || this.disposed) return;
      this.started = true;
      this.update();
    };
    const timer = setTimeout(go, LAYER_ZOOMS_WAIT_MS);
    void this.hub.layerZooms().then((layers) => {
      clearTimeout(timer);
      if (this.disposed) return;
      const c = this.config;
      this.zooms = dataZooms(layers, c.sourceLayer, c.filter, c.extrusion, this.minzoom!, this.maxzoom!);
      this.nothing = !this.zooms;
      if (this.started) this.update(); // came late: tiles of other zoom levels are given up
      else go();
    });
  }

  /** The tile workers (the source's, shared by its layers) with what each still has to answer. */
  get pool(): readonly PoolWorker[] {
    return this.hub?.pool ?? [];
  }

  /** Whether a tile of this zoom level can hold anything for the layer. */
  private hasData(z: number): boolean {
    return !this.nothing && (!this.zooms || (z >= this.zooms[0] && z <= this.zooms[1]));
  }

  /**
   * Load the ORIGINAL tiles covering the current view (clamped to the source's
   * zoom range) and retire tiles that left the view. Uses `coveringTiles` so above
   * `maxzoom` we fetch the canonical maxzoom tile (not MapLibre's overzoomed,
   * re-encoded one). Same tile URLs as MapLibre → deduped by the browser cache.
   *
   * In a pitched view MapLibre covers the distance with tiles of lower zoom
   * levels. Those the layer has nothing at are left out — the covering itself
   * stays the source's, so the layer still ends where its zoom levels end.
   */
  update() {
    if (!this.started || this.tileTemplate == null || this.minzoom == null || this.maxzoom == null) return;
    this.lastUpdate = performance.now();

    const wanted = new Set<string>();
    for (const id of this.map.coveringTiles({
      tileSize: 512,
      minzoom: this.minzoom,
      maxzoom: this.maxzoom,
      roundZoom: false,
      terrain: this.map.terrain,
    } as CoveringTilesOptions)) {
      const { x, y, z } = id.canonical;
      if (!this.hasData(z)) continue;
      const key = `${z}/${x}/${y}`;
      wanted.add(key);
      if (!this.tiles.has(key)) this.showTile(key, x, y, z);
    }

    for (const [key, entry] of this.tiles) {
      if (wanted.has(key)) continue;
      this.tiles.delete(key);
      this.retire(key, entry);
    }
  }

  /**
   * Load the tiles a camera would show, without showing them: they wait,
   * hidden, until the map gets there (see `kept`) and are then on screen without
   * loading. A further call replaces the view loaded ahead. Returns how many of
   * the view's tiles were not there yet; 0 also when nothing can be loaded ahead
   * (no tile cache, or MapLibre's internals this needs are not there).
   */
  preload(camera: PreloadCamera): number {
    if (!this.started || this.cacheBytes <= 0 || this.tileTemplate == null) return 0;
    let ids: { canonical: { x: number; y: number; z: number } }[];
    try {
      // MapLibre has no API for the tiles of a view the camera is not at: a
      // copy of its transform is set to that camera and handed to coveringTiles.
      const map = this.map as unknown as { transform?: { clone(): Transform }; _camera?: { transform?: { clone(): Transform } } };
      const tr = (map._camera?.transform ?? map.transform)!.clone();
      const c = camera.center;
      tr.setCenter(Array.isArray(c) ? new LngLat(c[0], c[1]) : new LngLat(c.lng, c.lat));
      tr.setZoom(camera.zoom);
      tr.setPitch(camera.pitch ?? 0);
      tr.setBearing(camera.bearing ?? 0);
      if (camera.padding) tr.setPadding({ top: 0, bottom: 0, left: 0, right: 0, ...camera.padding });
      ids = this.map.coveringTiles.call({ transform: tr, _camera: { transform: tr } } as unknown as Map, {
        tileSize: 512,
        minzoom: this.minzoom,
        maxzoom: this.maxzoom,
        roundZoom: false,
        terrain: this.map.terrain,
      } as CoveringTilesOptions);
    } catch {
      return 0;
    }
    const pins = new Set<string>();
    let asked = 0;
    for (const { canonical: { x, y, z } } of ids) {
      if (!this.hasData(z)) continue;
      const key = `${z}/${x}/${y}`;
      if (this.tiles.has(key) || this.emptyKeys.has(key)) continue;
      pins.add(key);
      if (this.kept.has(key)) continue;
      this.loadTile(key, x, y, z, this.kept);
      asked++;
    }
    this.pins = pins;
    this.trim();
    return asked;
  }

  /**
   * Whether the view is complete and at rest: every tile of it is loaded, has
   * its AO data (faded in), stands on the terrain as it is and has the colours
   * of the global state as it is, and no look at the view is due. Tiles loaded
   * ahead (see preload) do not count.
   */
  isIdle(): boolean {
    if (this.disposed || this.failed) return true;
    if (!this.started || this.moveEndTimer !== undefined || this.recolorRafId !== undefined) return false;
    if (this.rebakeRafId !== undefined || this.rebakeQueue.size > 0 || this.pendingDemRegions.length > 0) return false;
    for (const entry of this.tiles.values()) {
      if (entry.state === 'loading' || entry.aoDue || entry.aoSince !== undefined) return false;
    }
    return true;
  }

  /** How many bytes of tiles outside the view are kept (see `kept`); less gives up kept tiles at once. */
  setCacheBytes(bytes: number) {
    this.cacheBytes = Math.max(0, bytes);
    this.trim();
  }

  /** Give up every tile, shown and kept, and load the view again. */
  private reload() {
    for (const entry of this.tiles.values()) this.freeTile(entry);
    this.tiles.clear();
    for (const entry of this.kept.values()) this.freeTile(entry);
    this.kept.clear();
    this.keptBytes = 0;
    this.pins.clear();
    this.rebakeQueue.clear();
    this.update();
  }

  /** A tile enters the view: from the kept ones, known to be empty, or to be loaded. */
  private showTile(key: string, x: number, y: number, z: number) {
    const kept = this.kept.get(key);
    if (kept) {
      this.kept.delete(key);
      this.pins.delete(key);
      this.tiles.set(key, kept);
      if (kept.state !== 'ready') return; // loaded ahead and still on its way: shown when it arrives
      this.keptBytes -= kept.bytes ?? 0;
      if (this.map.terrain) {
        // Onto the terrain as this view loads it, so in the next frame: right now
        // the map still holds the terrain tiles of the view it comes from (coarser
        // ones for this place, as a rule), and tiles it takes from its cache
        // arrive without a `data` event.
        this.rebakeQueue.add(kept);
        this.scheduleRebake();
      } else if (kept.terrainEpoch !== this.terrainEpoch) {
        this.bake(kept);
      }
      if (kept.colorEpoch !== this.colorEpoch) this.recolor(kept);
      this.show(kept);
      this.map.triggerRepaint();
      return;
    }
    if (this.emptyKeys.has(key)) {
      const pow = 2 ** z;
      this.tiles.set(key, { state: 'empty', tileMx: x / pow, tileMy: y / pow, bounds: tileBounds(x, y, z) });
      return;
    }
    this.loadTile(key, x, y, z, this.tiles);
  }

  /** A tile left the view: kept (hidden) if it is ready and there is a tile cache, else given up. */
  private retire(key: string, entry: TileEntry) {
    const keep = this.cacheBytes > 0;
    if (entry.state === 'empty') {
      if (keep) this.rememberEmpty(key);
      return;
    }
    if (entry.state !== 'ready' || !keep) {
      this.freeTile(entry);
      return;
    }
    this.hide(entry);
    this.kept.set(key, entry);
    this.keptBytes += entry.bytes ?? 0;
    this.trim();
  }

  /**
   * Give up kept tiles, oldest first, until they fit `cacheBytes`. The view
   * loaded ahead stays even beyond that: it is about to be shown.
   */
  private trim() {
    if (this.keptBytes <= this.cacheBytes) return;
    for (const [key, entry] of this.kept) {
      if (entry.state !== 'ready' || this.pins.has(key)) continue;
      this.kept.delete(key);
      this.keptBytes -= entry.bytes ?? 0;
      this.freeTile(entry);
      if (this.keptBytes <= this.cacheBytes) return;
    }
  }

  private rememberEmpty(key: string) {
    if (this.emptyKeys.size >= EMPTY_KEYS_MAX) this.emptyKeys.clear();
    this.emptyKeys.add(key);
  }

  /** Put a ready tile's meshes into the scene / take them out (they keep their buffers). */
  private show(entry: TileEntry) {
    if (entry.mesh) this.group.add(entry.mesh);
    if (entry.skirtMesh) this.skirtGroup.add(entry.skirtMesh);
    this.placeMesh(entry);
  }

  private hide(entry: TileEntry) {
    if (entry.mesh) this.group.remove(entry.mesh);
    if (entry.skirtMesh) this.skirtGroup.remove(entry.skirtMesh);
  }

  /** Give a tile up for good: whatever of it is still on its way is called off, its buffers are freed. */
  private freeTile(entry: TileEntry) {
    if (entry.seq !== undefined) this.hub?.cancel(entry.seq);
    this.hide(entry);
    entry.geometry?.dispose();
    entry.skirtGeometry?.dispose();
    if (entry.ao) for (const t of Object.values(entry.ao)) t.dispose();
  }

  /** Bind a tile's AO tables to the shared material before its mesh draws. */
  private bindAo(entry: TileEntry) {
    const t = entry.ao ?? this.aoEmpty;
    const fade = entry.aoFade ?? 0;
    const u = this.material.uniforms;
    if (u.u_ao_faces.value === t.faces && u.u_ao_tile.value === fade) return;
    u.u_ao_tile.value = fade;
    u.u_ao_faces.value = t.faces;
    u.u_ao_cells.value = t.cells;
    u.u_ao_lists.value = t.lists;
    u.u_ao_edges.value = t.edges;
    this.material.uniformsNeedUpdate = true;
  }

  /** The tile's AO fade for its skirt, before the skirt draws. */
  private bindSkirt(entry: TileEntry) {
    const fade = entry.aoFade ?? 0;
    const u = this.skirtMaterial.uniforms;
    if (u.u_ao_tile.value === fade) return;
    u.u_ao_tile.value = fade;
    this.skirtMaterial.uniformsNeedUpdate = true;
  }

  /** A tile's ground skirt (not in the scene yet, see show). */
  private addSkirt(entry: TileEntry, skirt: SkirtResult) {
    const sg = new THREE.BufferGeometry();
    sg.setAttribute('position', new THREE.BufferAttribute(skirt.position, 3));
    sg.setAttribute('coord', uploadOnce(new THREE.BufferAttribute(skirt.coord, 2, true)));
    sg.setIndex(uploadOnce(new THREE.BufferAttribute(skirt.index, 1)));
    const skirtMesh = new THREE.Mesh(sg, this.skirtMaterial);
    skirtMesh.frustumCulled = true;
    skirtMesh.onBeforeRender = () => this.bindSkirt(entry);
    entry.skirtMesh = skirtMesh;
    entry.skirtGeometry = sg;
    entry.skirtColumn = skirt.column;
  }

  /** What a tile is to be made of, as its worker needs to know it. */
  private spec(): TileSpec {
    const c = this.config;
    const ao = this.aoEnabled;
    return {
      sourceLayer: c.sourceLayer,
      filter: c.filter,
      extrusion: c.extrusion,
      skirtRadius: ao ? c.ao.groundRadius : 0,
      ao,
      aoLater: ao, // The geometry first, the AO data in a second message
      aoRadius: c.ao.radius,
      prep: {
        color: c.colorSpec,
        opacity: c.opacitySpec,
        zoom: this.map.getZoom(),
        globalState: { ...globalStateOf(this.map) },
        flat: !this.map.terrain,
        properties: this.repaints,
      },
    };
  }

  /** Ask the hub for a canonical tile — for the view (`tiles`) or ahead of it (`kept`). */
  private loadTile(key: string, x: number, y: number, z: number, into: globalThis.Map<string, TileEntry>) {
    const pow = 2 ** z;
    const entry: TileEntry = {
      state: 'loading',
      tileMx: x / pow,
      tileMy: y / pow,
      bounds: tileBounds(x, y, z),
      colorEpoch: this.colorEpoch,
    };
    into.set(key, entry);
    const url = this.tileTemplate!
      .replace('{z}', String(z))
      .replace('{x}', String(x))
      .replace('{y}', String(y));
    const spec = this.spec();
    entry.aoLater = spec.ao && spec.aoLater === true;
    entry.seq = this.hub!.request(this.consumerId, key, x, y, z, url, spec);
  }

  onTileFailed(key: string, seq: number) {
    const shown = this.tiles.get(key);
    const entry = shown ?? this.kept.get(key);
    if (entry?.state !== 'loading' || entry.seq !== seq) return;
    entry.state = 'empty';
    if (!shown) this.kept.delete(key); // loaded ahead: the view asks again
  }

  onTileParsed(result: TileResult) {
    const shown = this.tiles.get(result.key);
    const entry = shown ?? this.kept.get(result.key);
    if (!entry || entry.state !== 'loading' || entry.seq !== result.seq) return; // given up mid-flight

    // The worker has done the per-vertex work (colours, on a map without terrain
    // the heights and the bounds): nothing of that here.
    const prepared = result.prepared;
    if (result.index.length === 0 || !prepared) {
      entry.state = 'empty';
      if (!shown) {
        this.kept.delete(result.key);
        this.rememberEmpty(result.key);
      }
      return;
    }

    entry.table = prepared.table;
    entry.properties = prepared.properties;
    // The base-relative Y, kept for re-baking (bake() overwrites it in `position`).
    entry.baseY = prepared.baseY;

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(result.position, 3));
    geometry.setAttribute('normal', uploadOnce(new THREE.BufferAttribute(result.normal, 3, true)));
    entry.geometry = geometry;
    if (entry.colorEpoch === this.colorEpoch) {
      geometry.setAttribute('color', uploadOnce(new THREE.BufferAttribute(prepared.color, 4, true)));
      entry.translucent = prepared.translucent;
    } else {
      this.recolor(entry); // the global state the worker's colours were made for has changed
    }
    // Integers, read as floats (not normalized). Without AO the worker sends
    // none (the shader then reads a constant and skips the AO anyway).
    if (result.aoh.length) {
      geometry.setAttribute('aoh', new THREE.BufferAttribute(result.aoh, 1));
      geometry.setAttribute('aoface', uploadOnce(new THREE.BufferAttribute(result.aoFace, 1)));
      geometry.setAttribute('aouv', uploadOnce(new THREE.BufferAttribute(result.aoUV, 2)));
    }
    if (result.aoTables) entry.ao = aoTextures(result.aoTables);
    if (result.aoh.length) entry.aoFade = 1; // AO data in one go: nothing to fade in
    else entry.aoDue = entry.aoLater;
    geometry.setIndex(uploadOnce(new THREE.BufferAttribute(result.index, 1)));

    const mesh = new THREE.Mesh(geometry, this.material);
    mesh.frustumCulled = true;
    mesh.onBeforeRender = () => this.bindAo(entry);

    if (result.skirt) this.addSkirt(entry, result.skirt);

    entry.state = 'ready';
    entry.tileScale = result.tileScale;
    entry.mesh = mesh;
    entry.column = result.column;
    entry.columnXZ = result.columnXZ;
    entry.anchors = result.anchors;

    if (prepared.flat && !this.map.terrain) {
      geometry.boundingSphere = toSphere(prepared.flat.sphere);
      if (entry.skirtGeometry && prepared.flat.skirtSphere) entry.skirtGeometry.boundingSphere = toSphere(prepared.flat.skirtSphere);
      entry.terrainEpoch = this.terrainEpoch;
      entry.bakedFlat = true;
    } else {
      this.bake(entry);
    }
    entry.bytes = bytesOf(entry);
    if (shown) {
      this.show(entry);
      this.map.triggerRepaint();
    } else {
      this.keptBytes += entry.bytes;
      this.trim();
    }
  }

  /**
   * The AO data of a tile that is on the map
   * already (or kept, hidden). Its vertices are the ones of the first pass (the
   * worker walks the tile again, for the AO), so the attributes go onto the
   * existing geometry; the AO then fades in (see syncAo).
   */
  onAoParsed(result: AoResult) {
    const shown = this.tiles.get(result.key);
    const entry = shown ?? this.kept.get(result.key);
    if (!entry || entry.state !== 'ready' || !entry.geometry || !entry.baseY || entry.seq !== result.seq) return; // given up meanwhile
    entry.aoDue = false;
    const vertexCount = entry.baseY.length;
    if (result.aoFace.length !== vertexCount || entry.aoFade !== undefined) {
      if (entry.aoFade === undefined) console.warn(`Fill3DLayer: AO data of tile ${result.key} does not fit its geometry — ignored`);
      return;
    }
    // On a map without terrain the worker has the heights above the ground already.
    const flatDone = result.aoh?.length === vertexCount && !this.map.terrain && entry.terrainEpoch === this.terrainEpoch;
    entry.geometry.setAttribute('aoh', new THREE.BufferAttribute(flatDone ? result.aoh! : new Int16Array(vertexCount), 1)); // else filled by bake
    entry.geometry.setAttribute('aoface', uploadOnce(new THREE.BufferAttribute(result.aoFace, 1)));
    entry.geometry.setAttribute('aouv', uploadOnce(new THREE.BufferAttribute(result.aoUV, 2)));
    if (result.aoTables) entry.ao = aoTextures(result.aoTables);
    // The columns now include the skirt's.
    entry.columnXZ = result.columnXZ;
    if (result.skirt) this.addSkirt(entry, result.skirt);
    entry.aoSince = performance.now();
    entry.aoFade = 0;

    if (flatDone) {
      if (entry.skirtGeometry && result.skirtSphere) entry.skirtGeometry.boundingSphere = toSphere(result.skirtSphere);
    } else {
      this.bake(entry);
    }
    const bytes = bytesOf(entry);
    if (shown) {
      this.show(entry);
      this.map.triggerRepaint();
    } else {
      this.keptBytes += bytes - (entry.bytes ?? 0);
    }
    entry.bytes = bytes;
    if (!shown) this.trim();
  }

  /**
   * Evaluate fill-3d-color / fill-3d-opacity per feature into the tile's per-vertex RGBA
   * attribute, here on the main thread: whenever the global state the paint reads has
   * changed (the worker does it for an arriving tile, see vertexColors). `translucent`: whether any vertex
   * got an alpha below 1 (a colour with alpha, a per-feature opacity) — such a tile needs
   * blending whatever the opacity uniform says.
   */
  private recolor(entry: TileEntry) {
    if (!entry.geometry || !entry.baseY || !entry.table) return;
    const { color, opacity } = this.config;
    const properties = entry.properties ?? NO_PROPERTIES;
    const colors = vertexColors(entry.baseY.length, entry.table, properties, color, opacity, this.map.getZoom(), globalStateOf(this.map));
    entry.geometry.setAttribute('color', uploadOnce(new THREE.BufferAttribute(colors.color, 4, true)));
    entry.translucent = colors.translucent;
    entry.colorEpoch = this.colorEpoch;
  }

  /**
   * Evaluate a feature-constant fill-3d-opacity for this frame (call each frame),
   * so zoom fades and global state apply smoothly. Returns the opacity in effect.
   *
   * Also decides whether this frame blends at all. A layer with a
   * `fill-3d-opacity` may be transparent, but while that opacity is 1 for every
   * feature (a pitch or zoom fade that has run its course) and no colour carries
   * an alpha, nothing is: the frame then draws like an opaque layer, in one pass
   * and writing its depth, without the depth pre-pass over all the geometry.
   */
  syncOpacity(): number {
    const { opacity } = this.config;
    const value = opacity.featureConstant
      ? opacity(this.map.getZoom(), {}, undefined, globalStateOf(this.map))
      : 1;
    this.material.uniforms.u_opacity.value = value;

    let blends = this.config.transparent && value < 1;
    if (this.config.transparent && !blends) {
      for (const entry of this.tiles.values()) {
        if (entry.translucent) { blends = true; break; }
      }
    }
    if (blends !== this.blends) {
      this.blends = blends;
      // Blending and depth writes are render state, not part of the program:
      // switching them needs no recompile.
      this.material.transparent = blends;
      this.material.depthWrite = !blends;
    }
    return value;
  }

  /** Re-bake the colours of the view's tiles after a global-state change (one per frame at most); kept tiles follow when they are shown. */
  private scheduleRecolor() {
    if (this.recolorRafId !== undefined) return;
    this.recolorRafId = requestAnimationFrame(() => {
      this.recolorRafId = undefined;
      this.colorEpoch++;
      for (const entry of this.tiles.values()) {
        if (entry.state === 'ready') this.recolor(entry);
      }
      this.map.triggerRepaint();
    });
  }

  /**
   * Whether this frame needs a depth pre-pass: only when it blends (see syncOpacity). The
   * pre-pass establishes the nearest surface per pixel; the colour pass then only
   * blends fragments at that depth — so interior/back walls of semi-transparent
   * buildings are hidden (like fill-extrusion's opaque-then-composite), while the
   * frontmost outer shell keeps its smooth per-feature alpha. An opaque layer
   * draws in one pass and writes its depth itself: a second pass over all the
   * geometry costs more than shading the fragments that end up hidden (measured
   * in Vienna on an Intel Arc: 3.9 instead of 5.0 ms for the layer).
   */
  get depthPrepass(): boolean {
    return this.blends;
  }

  /**
   * Set the scene up for one rendering pass. The depth pre-pass (see
   * {@link depthPrepass}) draws the buildings with a program that only places
   * the vertices.
   *
   * The skirts come after the buildings (they need their depth to hide behind
   * them, and write none): in the colour pass when they multiply, else in two
   * passes of their own (see {@link skirtPasses}).
   */
  setPass(pass: 'depth' | 'color' | 'skirtCollect' | 'skirtApply') {
    const skirtOnly = pass === 'skirtCollect' || pass === 'skirtApply';
    this.group.visible = !skirtOnly;
    this.scene.overrideMaterial = pass === 'depth' ? this.depthMaterial : null;
    const aoOn = this.material.uniforms.u_ao_intensity.value > 0;
    if (skirtOnly) {
      this.skirtGroup.visible = aoOn;
      setSkirtMode(this.skirtMaterial, pass === 'skirtCollect' ? 'collect' : 'apply');
    } else {
      this.skirtGroup.visible = pass === 'color' && aoOn && !this.config.skirtMax;
      setSkirtMode(this.skirtMaterial, 'multiply');
    }
  }

  /** Whether this frame needs the two skirt passes (strongest shade wins). */
  get skirtPasses(): boolean {
    return this.config.skirtMax && this.material.uniforms.u_ao_intensity.value > 0 && this.skirtGroup.children.length > 0;
  }

  /** Whether fake AO can be on at all (its intensity isn't a constant 0). */
  private get aoEnabled(): boolean {
    const i = this.config.ao.intensity;
    return !(i.featureConstant && i.globalStateRefs.size === 0 && i(0, {}) <= 0 && i(24, {}) <= 0);
  }

  /** Evaluate the fake-AO intensity for this frame (call each frame). */
  syncAo() {
    const value = Math.max(0, Math.min(1, this.config.ao.intensity(this.map.getZoom(), {}, undefined, globalStateOf(this.map))));
    this.material.uniforms.u_ao_intensity.value = value;
    this.skirtMaterial.uniforms.u_ao_intensity.value = value;
    this.skirtMaterial.uniforms.u_opacity.value = this.material.uniforms.u_opacity.value;

    // Tiles whose AO data has just arrived fade in.
    const now = performance.now();
    let fading = false;
    for (const entry of this.tiles.values()) {
      if (entry.aoSince === undefined) continue;
      entry.aoFade = Math.min(1, (now - entry.aoSince) / AO_FADE_MS);
      if (entry.aoFade < 1) fading = true;
      else entry.aoSince = undefined;
    }
    if (fading) this.map.triggerRepaint();
  }

  /**
   * Sync the shader light with the map's style light + bearing (call each frame).
   * Mirrors MapLibre's fill-extrusion: `sphericalToCartesian(light.position)` in
   * (east, south, up); for anchor 'viewport' rotate the horizontal part by the
   * bearing; then map into our (east, up, north) frame → (x, z, -y).
   */
  syncLight() {
    const light = this.map.getLight() as {
      anchor?: 'map' | 'viewport';
      position?: [number, number, number];
      color?: string;
      intensity?: number;
    };
    const [r, az, polar] = light.position ?? [1.15, 210, 30];
    const a = ((az + 90) * Math.PI) / 180;
    const p = (polar * Math.PI) / 180;
    let east = r * Math.cos(a) * Math.sin(p);
    let south = r * Math.sin(a) * Math.sin(p);
    const up = r * Math.cos(p);

    if ((light.anchor ?? 'viewport') === 'viewport') {
      const b = (this.map.getBearing() * Math.PI) / 180;
      const c = Math.cos(b);
      const s = Math.sin(b);
      [east, south] = [east * c - south * s, east * s + south * c];
    }

    const u = this.material.uniforms;
    (u.u_lightpos.value as THREE.Vector3).set(east, up, -south); // → (east, up, north)
    u.u_lightintensity.value = light.intensity ?? 0.5;
    // Read the light colour back in sRGB (MapLibre's u_lightcolor is sRGB).
    SCRATCH_COLOR.set(light.color ?? '#ffffff').getRGB(SCRATCH_SRGB, THREE.SRGBColorSpace);
    (u.u_lightcolor.value as THREE.Vector3).set(SCRATCH_SRGB.r, SCRATCH_SRGB.g, SCRATCH_SRGB.b);
  }

  /** Position + scale a tile mesh relative to the current scene origin. */
  private placeMesh(entry: TileEntry) {
    if (!entry.mesh || entry.tileScale === undefined) return;
    const o = this.origin;
    // Tile geometry is in metres at the tile's latitude; the projection matrix
    // works in metres at the origin's latitude → correct with a uniform scale.
    for (const mesh of [entry.mesh, entry.skirtMesh]) {
      if (!mesh) continue;
      mesh.scale.setScalar(entry.tileScale / o.s);
      mesh.position.set(
        (entry.tileMx - o.x) / o.s, // east metres from origin
        0,
        (o.y - entry.tileMy) / o.s, // north metres from origin
      );
    }
  }

  /** Re-anchor the origin to the map centre if it has drifted too far. */
  private maybeRebase() {
    const om = MercatorCoordinate.fromLngLat(this.map.getCenter(), 0);
    const dEast = (om.x - this.origin.x) / this.origin.s;
    const dNorth = (om.y - this.origin.y) / this.origin.s;
    if (Math.hypot(dEast, dNorth) <= REBASE_DISTANCE_M) return;

    this.origin = { x: om.x, y: om.y, z: om.z, s: om.meterInMercatorCoordinateUnits() };
    for (const entry of this.tiles.values()) this.placeMesh(entry);
  }

  /**
   * Add terrain elevation to each vertex Y: the height the building's base goes
   * to everywhere, except that vertices at its base follow the ground down where
   * it drops below that — so the downhill walls of a building on a slope reach
   * the ground instead of floating. (fill-extrusion sinks the base 10 m
   * instead; these meshes aren't hidden by terrain, so that would show.)
   *
   * Where the base goes: for a building in a group of touching buildings (the
   * source gives their heights in one frame) to its base height plus the group's
   * terrain fit (see anchorFits) — neighbours stay as they stand to each other in
   * the data; for any other building to the terrain height at its centre.
   *
   * The source itself sinks the walls standing on a building's base 10 m below
   * it as a reserve for slopes (buildings3d `MTK_BUILDINGS3D_GROUND_DEPTH`): the
   * only vertices below the base. The terrain surface covers that reserve. The
   * flat map covers nothing, so without terrain those vertices come up to the
   * base — else every wall shows 10 m too deep on a pitched map.
   *
   * `sampled`: the terrain heights the tile keeps (see TileEntry.groundCache)
   * are the terrain's as it is — terrainMoved has just sampled them — and are
   * used as they are.
   */
  private bake(entry: TileEntry, sampled = false) {
    if (!entry.geometry || !entry.baseY || !entry.table) return;
    entry.terrainEpoch = this.terrainEpoch;
    const table = entry.table;
    const attr = entry.geometry.getAttribute('position') as THREE.BufferAttribute;
    const pos = attr.array as Float32Array;
    const baseY = entry.baseY;
    const aoAttr = entry.geometry.getAttribute('aoh') as THREE.BufferAttribute | undefined;
    const ao = aoAttr?.array as Int16Array | undefined;
    const flat = !this.map.terrain;
    const elevationAt = createElevationSampler(this.map);
    const groundAt = createMercatorElevationSampler(this.map);
    const scale = entry.tileScale ?? 0;
    // The terrain under a vertex, sampled once per ground column: a wall's top
    // and bottom, the roof above it and the skirt at its foot share one.
    const column = entry.column!;
    const xz = entry.columnXZ!;
    // Kept as 32-bit floats and read back as such, so a bake from kept heights
    // gives what a bake from fresh ones does.
    let cache = entry.groundCache;
    let centres = entry.centreCache;
    if (!sampled || !cache || !centres || cache.length !== xz.length / 2) {
      cache = entry.groundCache = new Float32Array(xz.length / 2).fill(NaN);
      centres = entry.centreCache = new Float32Array(table.start.length).fill(NaN);
    }
    entry.bakedFlat = flat;
    const groundOf = (c: number) => {
      let g = cache[c];
      if (g !== g) {
        cache[c] = groundAt(entry.tileMx + xz[c * 2] * scale, entry.tileMy - xz[c * 2 + 1] * scale);
        g = cache[c];
      }
      return g;
    };
    // The terrain at a feature's centre (its anchor, for a rigid model).
    const centreOf = (i: number) => {
      let e = centres[i];
      if (e !== e) {
        centres[i] = elevationAt(table.lng[i], table.lat[i]);
        e = centres[i];
      }
      return e;
    };
    // Fake AO needs each vertex's height above the ground, but only where it can
    // reach: up to its radius, plus headroom for the ground falling away on a slope.
    const aoReach = this.aoEnabled ? this.config.ao.radius + 10 : -Infinity;
    const fits = flat || !scale || !entry.anchors ? undefined : anchorFits(entry.anchors, groundOf);
    // The flat map's ground under the vertices of rigid models.
    const rigidGround = flat && ao ? flatGroundOfRigid(table, column, baseY, xz.length / 2) : undefined;
    let moved = false;

    for (let i = 0; i < table.start.length; i++) {
      const start = table.start[i];
      const end = start + table.count[i];
      if (table.rigid[i]) {
        // A model built against the terrain around its anchor
        // (b.lng, b.lat) goes to the terrain height there as one rigid body, like a
        // GltfLayer model. Its feet reach below the ground on purpose (the generator's
        // skirt, lower ground than at the anchor): nothing follows the ground down,
        // and on the flat map nothing is clamped — what lies below it shows through.
        const ele = flat ? 0 : centreOf(i);
        for (let v = start; v < end; v++) {
          const rel = baseY[v];
          const y = rel + ele;
          if (pos[v * 3 + 1] !== Math.fround(y)) {
            pos[v * 3 + 1] = y;
            moved = true;
          }
          if (!ao) continue;
          const aboveGround = rigidGround ? rel - rigidGround[column[v]] : !flat && scale && rel < aoReach ? y - groundOf(column[v]) : rel;
          ao[v] = Math.max(-32766, Math.min(32766, Math.round(aboveGround * 10)));
        }
        continue;
      }
      const fit = fits && table.anchor[i] >= 0 ? fits[table.anchor[i]] : NaN;
      const ele = flat ? 0 : fit === fit ? table.base[i] + fit : centreOf(i);
      for (let v = start; v < end; v++) {
        const rel = baseY[v];
        let y: number;
        let aboveGround: number;
        if (flat) {
          y = Math.max(rel, 0);
          aboveGround = y;
        } else if (scale && (rel <= BASE_EPSILON_M || rel < aoReach)) {
          const ground = groundOf(column[v]);
          y = rel <= BASE_EPSILON_M ? Math.min(ele, ground) + rel : rel + ele;
          aboveGround = y - ground;
        } else {
          y = rel + ele;
          aboveGround = rel; // out of AO reach (or AO off): any value beyond the radius
        }
        if (pos[v * 3 + 1] !== Math.fround(y)) {
          pos[v * 3 + 1] = y;
          moved = true;
        }
        if (ao) ao[v] = Math.max(-32766, Math.min(32766, Math.round(aboveGround * 10)));
      }
    }
    // Heights that stayed as they were (a tile that stands already gets its AO data) are not uploaded again.
    if (moved || !entry.geometry.boundingSphere) {
      attr.needsUpdate = true;
      entry.geometry.boundingSphere = toSphere(boundingSphere(pos));
    }
    if (aoAttr) aoAttr.needsUpdate = true;

    // The skirt lies on the ground, lifted a little against z-fighting.
    if (entry.skirtGeometry && scale) {
      const sAttr = entry.skirtGeometry.getAttribute('position') as THREE.BufferAttribute;
      const sp = sAttr.array as Float32Array;
      const sc = entry.skirtColumn!;
      for (let i = 0; i < sc.length; i++) sp[i * 3 + 1] = (flat ? 0 : groundOf(sc[i])) + SKIRT_LIFT_M;
      sAttr.needsUpdate = true;
      entry.skirtGeometry.boundingSphere = toSphere(boundingSphere(sp));
    }
  }

  /** The terrain was switched on, off or re-set: every tile of the view is baked anew (see rebakeStep). */
  private rebakeAll() {
    this.pendingDemRegions = [];
    for (const entry of this.tiles.values()) {
      if (entry.state === 'ready') this.rebakeQueue.add(entry);
    }
    this.scheduleRebake();
  }

  /** Coalesce a burst of DEM `data` events into one look at the tiles on the next frame. */
  private scheduleRebake() {
    if (this.rebakeRafId !== undefined) return;
    this.rebakeRafId = requestAnimationFrame(() => this.rebakeStep());
  }

  /**
   * Bring the tiles onto changed terrain: those that overlap a DEM tile that
   * has arrived (and those shown again from the kept ones) are queued, and each
   * frame one of them is baked anew, the nearest to the map centre first — but
   * only if the terrain under it has changed at all (see terrainMoved). Baking
   * every overlapping tile in the frame the DEM tile arrived in blocked a map in
   * motion again and again (Vienna with terrain, 2026-11-18: 13 tiles in one
   * frame, 45 frames a tour with over 100 ms of baking), and most of those
   * bakes moved no vertex: coarser terrain tiles keep arriving for ground the
   * tile stands on in finer ones.
   */
  private rebakeStep() {
    this.rebakeRafId = undefined;
    const regions = this.pendingDemRegions;
    this.pendingDemRegions = [];
    if (regions.length) {
      this.terrainEpoch++; // kept tiles are checked when they are shown again
      for (const entry of this.tiles.values()) {
        if (entry.state !== 'ready') continue;
        if (regions.some((r) => overlaps(r, entry.bounds))) this.rebakeQueue.add(entry);
        else if (!this.rebakeQueue.has(entry)) entry.terrainEpoch = this.terrainEpoch; // the changed terrain is not under this tile
      }
    }
    if (!this.rebakeQueue.size) return;

    const centre = MercatorCoordinate.fromLngLat(this.map.getCenter(), 0);
    const t0 = performance.now();
    let baked = false;
    while (this.rebakeQueue.size && !baked && performance.now() - t0 < TERRAIN_CHECK_MS) {
      let next: TileEntry | undefined;
      let nearest = Infinity;
      for (const entry of this.rebakeQueue) {
        const d = Math.hypot(entry.tileMx - centre.x, entry.tileMy - centre.y);
        if (d < nearest) {
          nearest = d;
          next = entry;
        }
      }
      this.rebakeQueue.delete(next!);
      if (next!.state !== 'ready' || next!.mesh?.parent !== this.group) continue; // left the view meanwhile
      if (!this.terrainMoved(next!)) {
        next!.terrainEpoch = this.terrainEpoch;
        continue;
      }
      this.bake(next!, true);
      baked = true;
    }
    if (baked) this.map.triggerRepaint();
    if (this.rebakeQueue.size) this.scheduleRebake();
  }

  /**
   * Whether the terrain under a tile is another than the one it was last baked
   * onto: every height that bake took from the terrain (see
   * TileEntry.groundCache) is sampled again and compared, exactly. The samples
   * are kept — the bake that follows a `true` uses them (`sampled`). Costs a
   * quarter of a bake; a few hundred points per tile instead were not enough:
   * finer terrain data moves the ground by metres at an embankment and by
   * nothing a street away, and the buildings there kept standing on the old one.
   */
  private terrainMoved(entry: TileEntry): boolean {
    const flat = !this.map.terrain;
    const cache = entry.groundCache;
    const centres = entry.centreCache;
    const xz = entry.columnXZ;
    const table = entry.table;
    if (entry.bakedFlat !== flat) return true;
    if (flat) return false;
    if (!cache || !centres || !xz || !table || cache.length !== xz.length / 2) return true;
    const groundAt = createMercatorElevationSampler(this.map);
    const elevationAt = createElevationSampler(this.map);
    const scale = entry.tileScale ?? 0;
    let moved = false;
    for (let c = 0; c < cache.length; c++) {
      const was = cache[c];
      if (was !== was) continue; // not used
      cache[c] = groundAt(entry.tileMx + xz[c * 2] * scale, entry.tileMy - xz[c * 2 + 1] * scale);
      if (cache[c] !== was) moved = true;
    }
    for (let i = 0; i < centres.length; i++) {
      const was = centres[i];
      if (was !== was) continue;
      centres[i] = elevationAt(table.lng[i], table.lat[i]);
      if (centres[i] !== was) moved = true;
    }
    return moved;
  }

  /** Detach listeners, stop the worker and free all tile geometry. */
  dispose() {
    this.map.off('data', this.onData);
    this.map.off('moveend', this.onMoveEnd);
    this.map.off('move', this.onMove);
    this.map.off('terrain', this.onTerrain);
    this.map.off('webglcontextrestored', this.onContextRestored);
    if (this.onSourceData) this.map.off('sourcedata', this.onSourceData);
    if (this.rebakeRafId !== undefined) cancelAnimationFrame(this.rebakeRafId);
    if (this.recolorRafId !== undefined) cancelAnimationFrame(this.recolorRafId);
    if (this.moveEndTimer !== undefined) clearTimeout(this.moveEndTimer);
    this.unwatchGlobalState();
    this.disposed = true;
    for (const entry of this.tiles.values()) this.freeTile(entry);
    this.tiles.clear();
    for (const entry of this.kept.values()) this.freeTile(entry);
    this.kept.clear();
    this.rebakeQueue.clear();
    this.hub?.detach(this.consumerId);
    this.scene.remove(this.group);
    this.scene.remove(this.skirtGroup);
    this.material.dispose();
    this.depthMaterial.dispose();
    for (const t of Object.values(this.aoEmpty)) t.dispose();
    this.skirtMaterial.dispose();
  }
}

/**
 * Per group of touching buildings: how far the terrain lies above the source's
 * heights — the median of (terrain at a wall foot − the foot's height) over the
 * group's wall feet standing on the ground, weighted by the wall length each
 * stands for. The source's heights fit the terrain up to this offset (the
 * height datum) and what the terrain model misses of the ground's shape; the
 * median ignores feet where it misses most.
 */
function anchorFits(anchors: TerrainAnchors, groundOf: (column: number) => number): Float32Array {
  const groups = anchors.start.length - 1;
  const fits = new Float32Array(groups);
  const diff = new Float32Array(anchors.column.length);
  for (let i = 0; i < diff.length; i++) diff[i] = groundOf(anchors.column[i]) - anchors.height[i];
  let order = FIT_ORDER;
  for (let g = 0; g < groups; g++) {
    const start = anchors.start[g];
    const n = anchors.start[g + 1] - start;
    if (n > order.length) order = FIT_ORDER = new Uint32Array(n * 2);
    // The group's feet by their difference, ascending. Most groups have a
    // handful of feet: sorted by insertion (stable, like Array.sort); a long
    // row of houses goes through Array.sort.
    let total = 0;
    for (let k = 0; k < n; k++) total += anchors.weight[start + k];
    if (n <= FIT_INSERTION_MAX) {
      for (let k = 0; k < n; k++) {
        const i = start + k;
        const d = diff[i];
        let at = k;
        while (at > 0 && diff[order[at - 1]] > d) {
          order[at] = order[at - 1];
          at--;
        }
        order[at] = i;
      }
    } else {
      const sorted: number[] = [];
      for (let k = 0; k < n; k++) sorted.push(start + k);
      sorted.sort((a, b) => diff[a] - diff[b]);
      order.set(sorted);
    }
    let fit = NaN; // a group without feet: its buildings go to their centres
    let sum = 0;
    for (let k = 0; k < n; k++) {
      const i = order[k];
      sum += anchors.weight[i];
      if (sum >= total / 2) {
        fit = diff[i];
        break;
      }
    }
    fits[g] = fit;
  }
  return fits;
}

/** Scratch of anchorFits: a group's feet in order. */
let FIT_ORDER = new Uint32Array(256);
/** Groups of at most this many feet are sorted by insertion. */
const FIT_INSERTION_MAX = 32;

/** A bounding sphere the worker computed (x, y, z, radius). */
function toSphere(s: Float32Array): THREE.Sphere {
  return new THREE.Sphere(new THREE.Vector3(s[0], s[1], s[2]), s[3]);
}

const NO_PROPERTIES: Record<string, unknown>[] = new Proxy([], { get: () => EMPTY_PROPERTIES }) as never;
const EMPTY_PROPERTIES: Record<string, unknown> = {};

/** What a ready tile holds in memory: its vertex and index buffers, the skirt's, the AO tables, and what baking reads. */
function bytesOf(entry: TileEntry): number {
  let bytes = 0;
  for (const g of [entry.geometry, entry.skirtGeometry]) {
    if (!g) continue;
    for (const a of Object.values(g.attributes)) bytes += BYTES.get(a) ?? (a.array as ArrayBufferView).byteLength;
    if (g.index) bytes += BYTES.get(g.index) ?? (g.index.array as ArrayBufferView).byteLength;
  }
  if (entry.ao) for (const t of Object.values(entry.ao)) bytes += BYTES.get(t) ?? 0;
  for (const a of [entry.baseY, entry.column, entry.columnXZ, entry.skirtColumn, entry.groundCache, entry.centreCache]) bytes += a?.byteLength ?? 0;
  return bytes;
}

function toOrigin(lngLat: LngLat): SceneOrigin {
  const om = MercatorCoordinate.fromLngLat(lngLat, 0);
  return { x: om.x, y: om.y, z: om.z, s: om.meterInMercatorCoordinateUnits() };
}

/** Do two geographic bounding boxes intersect? (edge-touching counts.) */
function overlaps(a: GeoBounds, b: GeoBounds): boolean {
  return a.west <= b.east && a.east >= b.west && a.south <= b.north && a.north >= b.south;
}
