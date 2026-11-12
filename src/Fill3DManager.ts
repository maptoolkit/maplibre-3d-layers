import { MercatorCoordinate, LngLat } from 'maplibre-gl';
import type { CoveringTilesOptions, Map } from 'maplibre-gl';
import * as THREE from 'three';
import type { ElevationCache } from './ElevationCache';
import { createMercatorElevationSampler, tileBounds } from './ElevationCache';
import type { GeoBounds } from './ElevationCache';
import type { ExtrusionRequest, TileRequest, TileResult, FeatureSpan, AoTables, TerrainAnchors } from './mvt3dGeometry';
import { AO_CELL_M, AO_NO_FACE, AO_TEXTURE_SHIFT, AO_TEXTURE_WIDTH, AO_UV_UNIT_M } from './mvt3dGeometry';
import type { ColorEval, OpacityEval } from './paint';
import type { Evaluator } from './expression';
import { globalStateOf, watchGlobalState } from './globalState';
import type { FilterSpecification } from '@maplibre/maplibre-gl-style-spec';
// Inlined as a base64 blob so the built bundle stays single-file — no separate
// worker asset to host. Works for both the ESM package and the <script> build.
import Mvt3dWorker from './mvt3dWorker.ts?worker&inline';

/** The parts of MapLibre's `RequestParameters` a tile fetch uses. */
type TileRequestParameters = { url: string; credentials?: RequestCredentials; headers?: HeadersInit };

/** Vertices this close (metres) above a building's base count as its base. */
const BASE_EPSILON_M = 0.05;

/** Re-anchor the scene origin once the map centre drifts this far from it (metres). */
const REBASE_DISTANCE_M = 3000;

/** The skirt floats this far (metres) above the ground, against z-fighting with the terrain. */
const SKIRT_LIFT_M = 0.4;

/** Strength of the two fake-AO terms at intensity 1: darkening at the wall foot / in an inner corner. */
const AO_GROUND_WEIGHT = 0.6;
const AO_CORNER_WEIGHT = 0.5;

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
    if (u_ao_intensity > 0.0) {
      float ground = occlusion(max(v_aoh, 0.0));
      float corner = innerEdges();
      // The stronger term wins: an inner corner at the wall foot is no darker
      // than either alone (no doubling where edges meet).
      color.rgb *= 1.0 - u_ao_intensity * max(${AO_GROUND_WEIGHT.toFixed(2)} * ground, ${AO_CORNER_WEIGHT.toFixed(2)} * corner);
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
  uniform float u_opacity;
  uniform int u_mode; // 0 multiply, 1 collect the darkest shade in alpha, 2 apply it
  varying vec2 v_coord;
  void main() {
    float t = clamp(length(v_coord), 0.0, 1.0); // distance from the foot, in radii
    float shade = 1.0 - u_ao_intensity * ${AO_GROUND_WEIGHT.toFixed(2)} * u_opacity * (1.0 - t) * (1.0 - t);
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

function aoTextures(t: AoTables): AoTextures {
  return { faces: aoTexture(t.faces, 4), cells: aoTexture(t.cells, 1), lists: aoTexture(t.lists, 1), edges: aoTexture(t.edges, 4) };
}

/** Empty tables: integer samplers need integer textures bound, also for tiles without AO. */
function emptyAoTextures(): AoTextures {
  const row = (channels: number) => AO_TEXTURE_WIDTH * channels;
  return aoTextures({ faces: new Uint32Array(row(4)), cells: new Uint32Array(row(1)), lists: new Uint16Array(row(1)), edges: new Uint32Array(row(4)) });
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
  /** Whether opacity/color can be < 1 (→ transparent material). */
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
  /** Per vertex: the worker's y, the height above the feature base (no terrain
   *  yet). x and z never change, so only y is kept for re-baking. */
  baseY?: Float32Array;
  /** Per vertex / skirt vertex its ground column, per column x, z — and the
   *  ground height per column during a bake (NaN = not sampled yet). */
  column?: Uint32Array;
  skirtColumn?: Uint32Array;
  columnXZ?: Float32Array;
  groundCache?: Float32Array;
  features?: FeatureSpan[];
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
 * whenever new DEM tiles stream in (via the shared {@link ElevationCache}) or the
 * map's terrain is switched. Without terrain the meshes sit on the flat map,
 * nothing below it: the source sinks ground walls 10 m for slopes (see bake).
 */
export class Fill3DManager {
  /** Current scene anchor; read by the layer to build the projection matrix. */
  origin: SceneOrigin;

  private readonly map: Map;
  private readonly scene: THREE.Scene;
  private readonly group: THREE.Group;
  /** Ground skirts; drawn in the colour pass only, after the buildings' depth. */
  private readonly skirtGroup: THREE.Group;
  private readonly skirtMaterial: THREE.ShaderMaterial;
  private readonly elevation: ElevationCache;
  private readonly terrainSourceId: string;
  private readonly worker: Worker;
  private readonly tiles = new globalThis.Map<string, TileEntry>();
  private readonly material: THREE.ShaderMaterial;
  private readonly depthMaterial: THREE.ShaderMaterial;
  /** Empty AO tables, bound for tiles without inner edges. */
  private readonly aoEmpty: AoTextures;
  private readonly config: Fill3DConfig;
  private readonly onData: (ev: unknown) => void;
  private readonly onMoveEnd: () => void;
  private readonly onTerrain: () => void;
  private onSourceData?: (ev: unknown) => void;
  private readonly unwatchGlobalState: () => void;
  private recolorRafId?: number;

  // Terrain DEM tiles stream in as a burst of `data` events. Instead of a full
  // rebake per event, we accumulate the changed tiles' bounds and flush ONE
  // rebake on the next animation frame, touching only overlapping building tiles.
  private pendingDemRegions: GeoBounds[] = [];
  private rebakeRafId?: number;

  // Read from the MapLibre vector source once its TileJSON has loaded.
  private tileTemplate?: string;
  private minzoom?: number;
  private maxzoom?: number;

  constructor(
    map: Map,
    scene: THREE.Scene,
    sceneOrigin: LngLat,
    elevation: ElevationCache,
    terrainSourceId: string,
    config: Fill3DConfig,
  ) {
    this.map = map;
    this.scene = scene;
    this.elevation = elevation;
    this.terrainSourceId = terrainSourceId;
    this.config = config;
    this.origin = toOrigin(sceneOrigin);

    this.group = new THREE.Group();
    scene.add(this.group);
    this.skirtGroup = new THREE.Group();
    scene.add(this.skirtGroup);

    // Shades like MapLibre's fill-extrusion (flat directional ramp). Per-vertex
    // sRGB colour (fill-3d-color) + alpha (fill-3d-opacity) via the `color` attr.
    this.aoEmpty = emptyAoTextures();
    this.material = createFill3DMaterial(config.transparent, this.aoEmpty);
    this.depthMaterial = createDepthMaterial();
    this.material.uniforms.u_ao_radius.value = config.ao.radius;
    this.skirtMaterial = createSkirtMaterial();

    this.worker = new Mvt3dWorker();
    this.worker.onmessage = (e: MessageEvent<TileResult>) => this.onTileParsed(e.data);

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
      this.maybeRebase();
      this.update();
    };
    // Terrain switched on, off or re-set: the cached elevations are stale, and
    // without terrain the sunk wall bases must come up to the map (see bake).
    this.onTerrain = () => {
      this.elevation.clear();
      this.rebakeAll();
    };
    map.on('data', this.onData);
    map.on('moveend', this.onMoveEnd);
    map.on('terrain', this.onTerrain);

    // `["global-state", …]`: a per-feature value is baked into the vertices, so
    // re-bake it; a feature-constant opacity is a uniform read every frame.
    const { color, opacity } = config;
    const baked = [...color.globalStateRefs, ...(opacity.featureConstant ? [] : opacity.globalStateRefs)];
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
      return;
    }
    if (tryRead()) {
      this.update();
      return;
    }
    this.onSourceData = (ev) => {
      if ((ev as { sourceId?: string }).sourceId !== id) return;
      if (tryRead()) {
        this.map.off('sourcedata', this.onSourceData!);
        this.onSourceData = undefined;
        this.update();
      }
    };
    this.map.on('sourcedata', this.onSourceData);
  }

  /**
   * Load the ORIGINAL tiles covering the current view (clamped to the source's
   * zoom range) and drop tiles that left the view. Uses `coveringTiles` so above
   * `maxzoom` we fetch the canonical maxzoom tile (not MapLibre's overzoomed,
   * re-encoded one). Same tile URLs as MapLibre → deduped by the browser cache.
   */
  update() {
    if (this.tileTemplate == null || this.minzoom == null || this.maxzoom == null) return;

    const wanted = new Set<string>();
    for (const id of this.map.coveringTiles({
      tileSize: 512,
      minzoom: this.minzoom,
      maxzoom: this.maxzoom,
      roundZoom: false,
      terrain: this.map.terrain,
    } as CoveringTilesOptions)) {
      const { x, y, z } = id.canonical;
      const key = `${z}/${x}/${y}`;
      wanted.add(key);
      if (!this.tiles.has(key)) this.loadTile(key, x, y, z);
    }

    // Evict tiles no longer covered.
    for (const [key, entry] of this.tiles) {
      if (wanted.has(key)) continue;
      this.freeTile(entry);
      this.tiles.delete(key);
    }
  }

  private freeTile(entry: TileEntry) {
    if (entry.mesh) this.group.remove(entry.mesh);
    if (entry.skirtMesh) this.skirtGroup.remove(entry.skirtMesh);
    entry.geometry?.dispose();
    entry.skirtGeometry?.dispose();
    if (entry.ao) for (const t of Object.values(entry.ao)) t.dispose();
  }

  /** Bind a tile's AO tables to the shared material before its mesh draws. */
  private bindAo(entry: TileEntry) {
    const t = entry.ao ?? this.aoEmpty;
    const u = this.material.uniforms;
    if (u.u_ao_faces.value === t.faces) return;
    u.u_ao_faces.value = t.faces;
    u.u_ao_cells.value = t.cells;
    u.u_ao_lists.value = t.lists;
    u.u_ao_edges.value = t.edges;
    this.material.uniformsNeedUpdate = true;
  }

  /** Fetch a canonical tile's original pbf and hand it to the worker. */
  private loadTile(key: string, x: number, y: number, z: number) {
    const pow = 2 ** z;
    this.tiles.set(key, {
      state: 'loading',
      tileMx: x / pow,
      tileMy: y / pow,
      bounds: tileBounds(x, y, z),
    });

    const url = this.tileTemplate!
      .replace('{z}', String(z))
      .replace('{x}', String(x))
      .replace('{y}', String(y));

    this.tileRequest(url)
      .then((req) => fetch(req.url, { credentials: req.credentials, headers: req.headers }))
      .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(String(r.status)))))
      .then((buffer) => {
        const e = this.tiles.get(key);
        if (e?.state !== 'loading') return; // evicted mid-flight
        const req: TileRequest = {
          key,
          x,
          y,
          z,
          buffer,
          sourceLayer: this.config.sourceLayer,
          filter: this.config.filter,
          extrusion: this.config.extrusion,
          skirtRadius: this.aoEnabled ? this.config.ao.groundRadius : 0,
          ao: this.aoEnabled,
          aoRadius: this.config.ao.radius,
        };
        this.worker.postMessage(req, [buffer]);
      })
      .catch(() => {
        const entry = this.tiles.get(key);
        if (entry) entry.state = 'empty';
      });
  }

  /**
   * A tile request as MapLibre sends its own: through the map's `transformRequest`
   * (rewritten URL, credentials such as a session cookie, headers). `_requestManager`
   * is MapLibre-internal; without it, or if the transform fails, the URL is fetched as is.
   */
  private async tileRequest(url: string): Promise<TileRequestParameters> {
    const manager = (this.map as unknown as {
      _requestManager?: { transformRequest(url: string, type: string): TileRequestParameters | Promise<TileRequestParameters> };
    })._requestManager;
    if (!manager) return { url };
    try {
      return await manager.transformRequest(url, 'Tile');
    } catch {
      return { url };
    }
  }

  private onTileParsed(result: TileResult) {
    const entry = this.tiles.get(result.key);
    if (!entry || entry.state !== 'loading') return; // evicted mid-flight

    if (result.index.length === 0) {
      entry.state = 'empty';
      return;
    }

    const vertexCount = result.position.length / 3;
    // Keep the base-relative Y for re-baking (bake() overwrites it in `position`).
    const baseY = new Float32Array(vertexCount);
    for (let v = 0; v < vertexCount; v++) baseY[v] = result.position[v * 3 + 1];

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(result.position, 3));
    geometry.setAttribute('normal', new THREE.BufferAttribute(result.normal, 3, true));
    geometry.setAttribute('color', this.colorAttribute(vertexCount, result.features));
    // Integers, read as floats (not normalized). Without AO the worker sends
    // none (the shader then reads a constant and skips the AO anyway).
    if (result.aoh.length) {
      geometry.setAttribute('aoh', new THREE.BufferAttribute(result.aoh, 1));
      geometry.setAttribute('aoface', new THREE.BufferAttribute(result.aoFace, 1));
      geometry.setAttribute('aouv', new THREE.BufferAttribute(result.aoUV, 2));
    }
    if (result.aoTables) entry.ao = aoTextures(result.aoTables);
    geometry.setIndex(new THREE.BufferAttribute(result.index, 1));

    const mesh = new THREE.Mesh(geometry, this.material);
    mesh.frustumCulled = true;
    mesh.onBeforeRender = () => this.bindAo(entry);
    this.group.add(mesh);

    if (result.skirt) {
      const sg = new THREE.BufferGeometry();
      sg.setAttribute('position', new THREE.BufferAttribute(result.skirt.position, 3));
      sg.setAttribute('coord', new THREE.BufferAttribute(result.skirt.coord, 2, true));
      sg.setIndex(new THREE.BufferAttribute(result.skirt.index, 1));
      const skirtMesh = new THREE.Mesh(sg, this.skirtMaterial);
      skirtMesh.frustumCulled = true;
      this.skirtGroup.add(skirtMesh);
      entry.skirtMesh = skirtMesh;
      entry.skirtGeometry = sg;
    }

    entry.state = 'ready';
    entry.tileScale = result.tileScale;
    entry.mesh = mesh;
    entry.geometry = geometry;
    entry.baseY = baseY;
    entry.column = result.column;
    entry.columnXZ = result.columnXZ;
    entry.skirtColumn = result.skirt?.column;
    entry.features = result.features;
    entry.anchors = result.anchors;

    this.placeMesh(entry);
    this.bake(entry);
    this.map.triggerRepaint();
  }

  /** Evaluate fill-3d-color / fill-3d-opacity per feature into a per-vertex RGBA attribute.
   *  Colours stay sRGB — the shader shades and outputs in sRGB, matching fill-extrusion.
   *  8 bits per channel (normalized), like the colour itself; a quarter of float32.
   *  A feature-constant opacity is left out here: it's the `u_opacity` uniform. */
  private colorAttribute(vertexCount: number, features: FeatureSpan[]): THREE.BufferAttribute {
    const zoom = this.map.getZoom();
    const state = globalStateOf(this.map);
    const { color, opacity } = this.config;
    const rgba = new Uint8Array(vertexCount * 4);
    for (const b of features) {
      const [r, g, bl, ca] = color(zoom, b.properties, undefined, state);
      const a = opacity.featureConstant ? ca : ca * opacity(zoom, b.properties, undefined, state);
      const R = Math.round(r * 255), G = Math.round(g * 255), B = Math.round(bl * 255), A = Math.round(a * 255);
      const end = b.start + b.count;
      for (let v = b.start; v < end; v++) {
        rgba[v * 4] = R;
        rgba[v * 4 + 1] = G;
        rgba[v * 4 + 2] = B;
        rgba[v * 4 + 3] = A;
      }
    }
    return new THREE.BufferAttribute(rgba, 4, true);
  }

  /**
   * Evaluate a feature-constant fill-3d-opacity for this frame (call each frame),
   * so zoom fades and global state apply smoothly. Returns the opacity in effect.
   */
  syncOpacity(): number {
    const { opacity } = this.config;
    const value = opacity.featureConstant
      ? opacity(this.map.getZoom(), {}, undefined, globalStateOf(this.map))
      : 1;
    this.material.uniforms.u_opacity.value = value;
    return value;
  }

  /** Re-bake every tile's colours after a global-state change (one per frame at most). */
  private scheduleRecolor() {
    if (this.recolorRafId !== undefined) return;
    this.recolorRafId = requestAnimationFrame(() => {
      this.recolorRafId = undefined;
      for (const entry of this.tiles.values()) {
        if (entry.state !== 'ready' || !entry.geometry || !entry.baseY || !entry.features) continue;
        entry.geometry.setAttribute('color', this.colorAttribute(entry.baseY.length, entry.features));
      }
      this.map.triggerRepaint();
    });
  }

  /**
   * Whether the layer needs a depth pre-pass: only when it is transparent. The
   * pre-pass establishes the nearest surface per pixel; the colour pass then only
   * blends fragments at that depth — so interior/back walls of semi-transparent
   * buildings are hidden (like fill-extrusion's opaque-then-composite), while the
   * frontmost outer shell keeps its smooth per-feature alpha. An opaque layer
   * draws in one pass and writes its depth itself: a second pass over all the
   * geometry costs more than shading the fragments that end up hidden (measured
   * in Vienna on an Intel Arc: 3.9 instead of 5.0 ms for the layer).
   */
  get depthPrepass(): boolean {
    return this.config.transparent;
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
   */
  private bake(entry: TileEntry) {
    if (!entry.geometry || !entry.baseY || !entry.features) return;
    const attr = entry.geometry.getAttribute('position') as THREE.BufferAttribute;
    const pos = attr.array as Float32Array;
    const baseY = entry.baseY;
    const aoAttr = entry.geometry.getAttribute('aoh') as THREE.BufferAttribute | undefined;
    const ao = aoAttr?.array as Int16Array | undefined;
    const flat = !this.map.terrain;
    const elevationAt = this.elevation.batch();
    const groundAt = createMercatorElevationSampler(this.map);
    const scale = entry.tileScale ?? 0;
    // The terrain under a vertex, sampled once per ground column: a wall's top
    // and bottom, the roof above it and the skirt at its foot share one.
    const column = entry.column!;
    const xz = entry.columnXZ!;
    const cache = (entry.groundCache ??= new Float32Array(xz.length / 2));
    cache.fill(NaN);
    const groundOf = (c: number) => {
      let g = cache[c];
      if (g !== g) cache[c] = g = groundAt(entry.tileMx + xz[c * 2] * scale, entry.tileMy - xz[c * 2 + 1] * scale);
      return g;
    };
    // Fake AO needs each vertex's height above the ground, but only where it can
    // reach: up to its radius, plus headroom for the ground falling away on a slope.
    const aoReach = this.aoEnabled ? this.config.ao.radius + 10 : -Infinity;
    const fits = flat || !scale || !entry.anchors ? undefined : anchorFits(entry.anchors, groundOf);

    for (const b of entry.features) {
      const fit = fits && b.anchor !== undefined ? fits[b.anchor] : NaN;
      const ele = flat ? 0 : fit === fit ? b.base! + fit : elevationAt(new LngLat(b.lng, b.lat));
      const end = b.start + b.count;
      for (let v = b.start; v < end; v++) {
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
        pos[v * 3 + 1] = y;
        if (ao) ao[v] = Math.max(-32766, Math.min(32766, Math.round(aboveGround * 10)));
      }
    }
    attr.needsUpdate = true;
    if (aoAttr) aoAttr.needsUpdate = true;
    entry.geometry.computeBoundingSphere();

    // The skirt lies on the ground, lifted a little against z-fighting.
    if (entry.skirtGeometry && scale) {
      const sAttr = entry.skirtGeometry.getAttribute('position') as THREE.BufferAttribute;
      const sp = sAttr.array as Float32Array;
      const sc = entry.skirtColumn!;
      for (let i = 0; i < sc.length; i++) sp[i * 3 + 1] = (flat ? 0 : groundOf(sc[i])) + SKIRT_LIFT_M;
      sAttr.needsUpdate = true;
      entry.skirtGeometry.computeBoundingSphere();
    }
  }

  /** Rebake every ready tile against the map's current terrain (or none). */
  private rebakeAll() {
    this.pendingDemRegions = [];
    let changed = 0;
    for (const entry of this.tiles.values()) {
      if (entry.state !== 'ready') continue;
      this.bake(entry);
      changed++;
    }
    if (changed) this.map.triggerRepaint();
  }

  /** Coalesce a burst of DEM `data` events into one rebake on the next frame. */
  private scheduleRebake() {
    if (this.rebakeRafId !== undefined) return;
    this.rebakeRafId = requestAnimationFrame(() => {
      this.rebakeRafId = undefined;
      this.flushRebake();
    });
  }

  /** Rebake only the ready tiles that overlap a DEM tile changed this frame. */
  private flushRebake() {
    const regions = this.pendingDemRegions;
    this.pendingDemRegions = [];
    if (regions.length === 0) return;

    let changed = 0;
    for (const entry of this.tiles.values()) {
      if (entry.state !== 'ready') continue;
      if (!regions.some((r) => overlaps(r, entry.bounds))) continue;
      this.bake(entry);
      changed++;
    }
    if (changed) this.map.triggerRepaint();
  }

  /** Detach listeners, stop the worker and free all tile geometry. */
  dispose() {
    this.map.off('data', this.onData);
    this.map.off('moveend', this.onMoveEnd);
    this.map.off('terrain', this.onTerrain);
    if (this.onSourceData) this.map.off('sourcedata', this.onSourceData);
    if (this.rebakeRafId !== undefined) cancelAnimationFrame(this.rebakeRafId);
    if (this.recolorRafId !== undefined) cancelAnimationFrame(this.recolorRafId);
    this.unwatchGlobalState();
    this.worker.terminate();
    for (const entry of this.tiles.values()) this.freeTile(entry);
    this.tiles.clear();
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
  const order: number[] = [];
  for (let g = 0; g < groups; g++) {
    order.length = 0;
    let total = 0;
    for (let i = anchors.start[g]; i < anchors.start[g + 1]; i++) {
      order.push(i);
      total += anchors.weight[i];
    }
    order.sort((a, b) => diff[a] - diff[b]);
    let fit = NaN; // a group without feet: its buildings go to their centres
    let sum = 0;
    for (const i of order) {
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

function toOrigin(lngLat: LngLat): SceneOrigin {
  const om = MercatorCoordinate.fromLngLat(lngLat, 0);
  return { x: om.x, y: om.y, z: om.z, s: om.meterInMercatorCoordinateUnits() };
}

/** Do two geographic bounding boxes intersect? (edge-touching counts.) */
function overlaps(a: GeoBounds, b: GeoBounds): boolean {
  return a.west <= b.east && a.east >= b.west && a.south <= b.north && a.north >= b.south;
}
