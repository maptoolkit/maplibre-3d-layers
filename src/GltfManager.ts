import { LngLat, MercatorCoordinate } from 'maplibre-gl';
import type { Map } from 'maplibre-gl';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { featureFilter } from '@maplibre/maplibre-gl-style-spec';
import type { FilterSpecification, GlobalProperties } from '@maplibre/maplibre-gl-style-spec';
import { createElevationSampler, tileBounds } from './ElevationCache';
import type { GeoBounds } from './ElevationCache';
import type { Evaluator, GlobalState, Rgba, Vec3 } from './expression';
import { globalStateOf, watchGlobalState } from './globalState';
import type { Placement } from './GltfLayer';
import {
  fromUnitX,
  fromUnitY,
  hash32,
  lineLength,
  mercatorScale,
  placeAlongLine,
  polygonArea,
  positionKey,
  scatterInPolygon,
  toMercX,
  toMercY,
  unitX,
  unitY,
} from './placement';
import type { Accept, MercBounds, PlacedPoint, RowSpan } from './placement';

const D2R = Math.PI / 180;

/** Viewport padding (fraction of the view size) when culling source features. */
const CULL_PAD = 0.25;

/**
 * Screen margin (fraction of the size) for the on-screen test, so panning and
 * rotating don't pop models at the edges. None at the top: on a pitched map
 * the top edge already looks toward the horizon, and a margin there reaches
 * kilometres further — it held ~40% of the models in a 60° test view.
 */
const FRAME_PAD = 0.1;

/** Smallest instance capacity to allocate — avoids churn on tiny feature sets. */
const MIN_CAPACITY = 16;

/** Source + style config for the instanced models. */
export interface GltfConfig {
  /** Id of the MapLibre source to read point features from. */
  sourceId: string;
  /** Vector-tile layer name — required for vector sources, ignored for geojson. */
  sourceLayer?: string;
  /** MapLibre filter expression; features that fail are not rendered. */
  filter?: FilterSpecification;
  /** Model id → glTF/glb URL. */
  models: Record<string, string>;
  /** Which model a feature uses (`layout['model-id']`). */
  modelId: Evaluator<string>;
  /** Degrees around [east, north, up] (`paint['model-rotation']`). */
  rotation: Evaluator<Vec3>;
  /** Per-axis factor (`paint['model-scale']`). */
  scale: Evaluator<Vec3>;
  /** Metres along [east, north, up] (`paint['model-translation']`). */
  translation: Evaluator<Vec3>;
  /** Multiplied onto the model's own material colour (`paint['model-color']`). */
  color?: Evaluator<Rgba>;
  /** Multiplied onto the model's own alpha (`paint['model-opacity']`). */
  opacity?: Evaluator<number>;
  /** How features become instances (`layout['model-placement']`). */
  placement: Placement;
  /** Metres between generated instances (`layout['model-spacing']`). */
  spacing: Evaluator<number>;
  /** Full-density radius around the camera, in metres (`layout['model-thinning-distance']`). */
  thinningDistance: Evaluator<number>;
  /** Metres from the camera beyond which models draw as billboards (`layout['model-impostor-distance']`). */
  impostorDistance: Evaluator<number>;
  /** Upper bound on instances per build; denser data is thinned to fit. */
  maxInstances: number;
  /** Override the models' `doubleSided` flag; undefined keeps it. */
  doubleSided?: boolean;
  /** Layer is hidden below this zoom (MapLibre semantics). */
  minzoom?: number;
  /** Layer is hidden at and above this zoom (MapLibre semantics). */
  maxzoom?: number;
}

/** Smallest spacing we honour — also keeps the placement grid keys collision-free. */
const MIN_SPACING_M = 1;

/** Safety net if the pre-pass estimate was too optimistic. */
const HARD_STOP_FACTOR = 1.2;

/** Beyond this many halvings (density < 1/64) distance thinning places nothing. */
const THIN_HALVINGS = 6;

/** A mesh counts as closed (front faces suffice) up to this share of open edges. */
const MAX_OPEN_EDGE_SHARE = 0.1;

/** Quiet time (ms) after the last source tile before a data-driven rebuild. */
const DATA_SETTLE_MS = 150;

/** Headroom on the last build's thinning scale when pre-filtering candidates. */
const THIN_HINT_SLACK = 1.5;

/** Collect up to this many x `maxInstances` candidates before selecting. */
const CANDIDATE_FACTOR = 20;

/** Scene anchor in mercator + the metric at its latitude. */
export interface SceneOrigin {
  x: number;
  y: number;
  z: number;
  s: number;
}

/** One drawable primitive of a loaded glTF, with its transform inside the file. */
interface Proto {
  geometry: THREE.BufferGeometry;
  material: THREE.Material | THREE.Material[];
  matrix: THREE.Matrix4;
}

interface ModelEntry {
  url: string;
  state: 'loading' | 'ready' | 'error';
  protos: Proto[];
  /** An impostor's baked texture (its only proto is the billboard quad). */
  target?: THREE.WebGLRenderTarget;
}

/** Light for baking impostors, in the bake frame (x=east, y=up, z=toward the viewer). */
export interface ImpostorLight {
  direction: THREE.Vector3;
  color: THREE.Color;
  ambient: number;
  directional: number;
  environment: THREE.Texture | null;
  environmentIntensity: number;
}

/** Key of a model's impostor in `models` — can't collide with a user model id. */
const impostorKey = (modelId: string) => `\u0000impostor:${modelId}`;

/** Impostor texture width in pixels; the height follows the model's aspect. */
const IMPOSTOR_RES = 128;

// The bake is cleared to transparent black, so edge texels (and every mip
// level) hold colour x coverage. Undo that, or distant, heavily minified
// impostors darken toward black.
const IMPOSTOR_MAP_FRAGMENT = /* glsl */ `
#include <map_fragment>
#ifdef USE_MAP
  diffuseColor.rgb /= max( sampledDiffuseColor.a, 0.0001 );
#endif
`;

/** Degrees above the horizon that impostors are baked from. */
const IMPOSTOR_ELEVATION = 20;

// Cylindrical billboard: the quad turns about the vertical axis to face the
// camera (mtkRight = the view's horizontal right vector, in the instance frame)
// and keeps its base on the ground. Only the instance's position and per-axis
// scale are used, so the instance matrix can stay the full-model one.
const IMPOSTOR_PARS_VERTEX = /* glsl */ `
uniform vec3 mtkRight;
#include <common>
`;
const IMPOSTOR_PROJECT_VERTEX = /* glsl */ `
vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_INSTANCING
  mvPosition = vec4(
    instanceMatrix[ 3 ].xyz
      + mtkRight * transformed.x * length( instanceMatrix[ 0 ].xyz )
      + vec3( 0.0, transformed.y * length( instanceMatrix[ 1 ].xyz ), 0.0 ),
    1.0
  );
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;
`;

/**
 * A source feature kept in its ORIGINAL geometry. Expansion into instances
 * happens in `build()`, not here, because `model-spacing` is a style property
 * and needs the current zoom plus the feature's properties to evaluate.
 */
interface RawFeature {
  geometry: GeoJSONGeometry;
  bbox: GeoBounds;
  props: Record<string, unknown>;
  id?: string | number;
}

/** Everything needed to (re)compute one instance's matrix. */
interface Rec {
  lng: number;
  lat: number;
  /** Metres east/north of the scene origin. */
  east: number;
  north: number;
  rot: Vec3;
  scl: Vec3;
  tr: Vec3;
  /** The feature it came from — re-evaluates colour/alpha on a global-state change. */
  props: Record<string, unknown>;
  id?: string | number;
}

/**
 * All instances of ONE model id: one InstancedMesh per glTF primitive.
 *
 * Buffers are allocated to a power-of-two `capacity` and only the first
 * `recs.length` instances are drawn, so a feed whose feature count wobbles
 * rewrites existing buffers instead of reallocating GPU memory every update.
 */
interface Batch {
  meshes: THREE.InstancedMesh[];
  protos: Proto[];
  capacity: number;
  recs: Rec[];
  bounds: GeoBounds;
  alphaAttr?: THREE.InstancedBufferAttribute;
}

// Patches three's built-in materials so per-instance colour AND per-instance
// alpha reach the fragment shader. three writes `instanceColor` into `vColor`
// in the vertex stage but only *consumes* vColor when USE_COLOR(_ALPHA) is set
// (i.e. when the geometry has a vertex-colour attribute), so an unmodified
// glTF material ignores setColorAt(). `instanceAlpha` is ours entirely.
const PARS_VERTEX = /* glsl */ `
#include <color_pars_vertex>
#ifdef MTK_INSTANCE_ALPHA
  attribute float instanceAlpha;
  varying float vInstanceAlpha;
#endif
`;

const BODY_VERTEX = /* glsl */ `
#include <color_vertex>
#ifdef MTK_INSTANCE_ALPHA
  vInstanceAlpha = instanceAlpha;
#endif
`;

const PARS_FRAGMENT = /* glsl */ `
#include <color_pars_fragment>
#if defined( USE_INSTANCING_COLOR ) && !defined( USE_COLOR ) && !defined( USE_COLOR_ALPHA )
  varying vec4 vColor;
#endif
#ifdef MTK_INSTANCE_ALPHA
  varying float vInstanceAlpha;
#endif
#ifdef MTK_OPACITY
  uniform float mtkOpacity;
#endif
`;

const BODY_FRAGMENT = /* glsl */ `
#include <color_fragment>
#if defined( USE_INSTANCING_COLOR ) && !defined( USE_COLOR ) && !defined( USE_COLOR_ALPHA )
  diffuseColor.rgb *= vColor.rgb;
#endif
#ifdef MTK_INSTANCE_ALPHA
  diffuseColor.a *= vInstanceAlpha;
#endif
#ifdef MTK_OPACITY
  diffuseColor.a *= mtkOpacity;
#endif
`;

/**
 * Reads point features from a MapLibre source, groups them by `model-id` and
 * draws each group as InstancedMeshes of a loaded glTF — one instance per
 * feature, one draw call per glTF primitive (so thousands of features stay cheap).
 *
 * Instance positions are metres relative to {@link origin}, which is re-anchored
 * to the map centre on every rebuild — that keeps float32 precision high anywhere
 * on earth. Vertical placement is sampled from the map's terrain and re-baked
 * whenever new DEM tiles stream in.
 *
 * All work is coalesced into one animation frame: a `setData()` on a geojson
 * source emits a burst of `data` events, and each one must not trigger its own
 * read + rebuild.
 */
export class GltfManager {
  /** Current scene anchor; read by the layer to build the projection matrix. */
  origin: SceneOrigin;

  private readonly map: Map;
  private readonly group: THREE.Group;
  private readonly scene: THREE.Scene;
  private readonly terrainSourceId: string;
  private readonly config: GltfConfig;
  private readonly models = new globalThis.Map<string, ModelEntry>();
  private readonly batches = new globalThis.Map<string, Batch>();
  private readonly loader = new GLTFLoader();
  private readonly filter: (g: GlobalProperties, f: unknown) => boolean;
  private readonly warned = new Set<string>();
  /** glTF files share one material across meshes — patch each exactly once. */
  private readonly patched = new WeakSet<THREE.Material>();
  /** View's horizontal right vector in the instance frame, shared by all impostor shaders. */
  private readonly impostorRight = { value: new THREE.Vector3(1, 0, 0) };
  /** The distance-thinning scale the last build fitted; seeds the next one's cutoff. */
  private lastThinScale = 1;
  /** Whether a material was transparent as authored (e.g. glTF alphaMode BLEND). */
  private readonly ownTransparent = new WeakMap<THREE.Material, boolean>();
  /** `model-opacity` when it's the same for every instance, shared by all shaders. */
  private readonly opacityUniform = { value: 1 };
  /** Some instance alpha < 1 (per-instance opacity) / the uniform opacity < 1. */
  private instanceFading = false;
  private uniformFading = false;

  /** Cached source features; re-read on source data, re-culled on move. */
  private features: RawFeature[] = [];
  /** geojson sources deliver the whole dataset at once — vector ones per tile. */
  private readonly wholeDataset: boolean;

  private readonly onData: (ev: unknown) => void;
  private readonly onMoveEnd: () => void;
  private readonly unwatchGlobalState: (() => void)[];
  private paintRafId?: number;
  private needsRead = false;
  private needsBuild = false;
  private pendingDemRegions: GeoBounds[] = [];
  private rafId?: number;
  private settleTimer?: ReturnType<typeof setTimeout>;
  private flushing = false;
  private viewMoved = true; // since the last read/build
  private readonly onMove = () => (this.viewMoved = true);
  private disposed = false;

  // Scratch objects reused while writing instance matrices.
  private readonly sPos = new THREE.Vector3();
  private readonly sQuat = new THREE.Quaternion();
  private readonly sEuler = new THREE.Euler();
  private readonly sScale = new THREE.Vector3();
  private readonly sBase = new THREE.Matrix4();
  private readonly sMat = new THREE.Matrix4();
  private readonly sColor = new THREE.Color();

  constructor(
    map: Map,
    scene: THREE.Scene,
    sceneOrigin: LngLat,
    terrainSourceId: string,
    config: GltfConfig,
  ) {
    this.map = map;
    this.scene = scene;
    this.terrainSourceId = terrainSourceId;
    this.config = config;
    this.origin = toOrigin(sceneOrigin);
    this.filter = featureFilter(config.filter, 'filter').filter as typeof this.filter;

    this.group = new THREE.Group();
    scene.add(this.group);

    const src = map.getSource(config.sourceId);
    if (!src) {
      console.error(`GltfLayer: source '${config.sourceId}' not found — add it before the layer`);
    }
    this.wholeDataset = src?.type === 'geojson';

    for (const [id, url] of Object.entries(config.models)) this.loadModel(id, url);

    this.onData = (ev) => {
      const e = ev as {
        sourceId?: string;
        tile?: { tileID?: { canonical?: { x: number; y: number; z: number } } };
      };
      if (e.sourceId === this.terrainSourceId && e.tile?.tileID?.canonical) {
        const c = e.tile.tileID.canonical;
        this.pendingDemRegions.push(tileBounds(c.x, c.y, c.z));
        this.schedule();
      } else if (e.sourceId === this.config.sourceId) {
        this.refreshSoon();
      }
    };
    // A whole-dataset source only needs re-culling when the view moves; a tiled
    // one has to be re-queried, because its result IS the viewport.
    this.onMoveEnd = () => (this.wholeDataset ? this.schedule(false, true) : this.refresh());
    map.on('data', this.onData);
    map.on('moveend', this.onMoveEnd);
    map.on('move', this.onMove);

    // `["global-state", …]`. Colour/alpha are rewritten in place; anything that
    // decides placement or transforms needs a rebuild, which a move does anyway
    // on moveend. A uniform opacity is read every frame, so it only repaints.
    const refs = (...evals: (Evaluator<unknown> | undefined)[]) =>
      evals.flatMap((e) => (e ? [...e.globalStateRefs] : []));
    const { color, opacity } = config;
    const whenStill = (fn: () => void) => () => {
      if (!map.isMoving()) fn();
    };
    this.unwatchGlobalState = [
      watchGlobalState(map, refs(color, this.instanceOpacity ? opacity : undefined), () => this.schedulePaint()),
      watchGlobalState(map, refs(opacity), () => map.triggerRepaint()),
      watchGlobalState(
        map,
        refs(config.modelId, config.rotation, config.scale, config.translation, config.spacing,
          config.thinningDistance, config.impostorDistance),
        whenStill(() => this.schedule(false, true)),
      ),
      watchGlobalState(
        map,
        featureFilter(config.filter, 'filter').getGlobalStateRefs(),
        whenStill(() => this.refresh()),
      ),
    ];

    this.refresh();
  }

  /** `model-opacity` differs per instance → per-instance alpha attribute. */
  private get instanceOpacity(): boolean {
    return !!this.config.opacity && !this.config.opacity.featureConstant;
  }

  /** `model-opacity` is the same for all instances → one uniform, set each frame. */
  private get uniformOpacity(): boolean {
    return !!this.config.opacity?.featureConstant;
  }

  /**
   * Evaluate a feature-constant `model-opacity` for this frame (call each
   * frame), so zoom fades and global state apply smoothly. Returns the opacity
   * in effect.
   */
  syncOpacity(): number {
    const { opacity } = this.config;
    if (!opacity?.featureConstant) return 1;
    const value = opacity(this.map.getZoom(), {}, undefined, globalStateOf(this.map));
    this.opacityUniform.value = value;
    if (value < 1 !== this.uniformFading) {
      this.uniformFading = value < 1;
      this.setFading(this.instanceFading || this.uniformFading);
    }
    return value;
  }

  /**
   * Refresh for new source data. Tiles arrive spread over many frames, and a
   * rebuild per tile costs about as much as one for all of them — so let a
   * burst settle first, and don't rebuild mid-move (moveend refreshes anyway).
   */
  private refreshSoon() {
    if (this.disposed) return;
    if (this.settleTimer !== undefined) clearTimeout(this.settleTimer);
    this.settleTimer = setTimeout(() => {
      this.settleTimer = undefined;
      if (!this.map.isMoving()) this.refresh();
    }, DATA_SETTLE_MS);
  }

  /**
   * What the end of a move or a data burst does, right away (also mid-move, without waiting for
   * the data to settle). Nothing if neither happened since the last read/build.
   */
  updateNow() {
    if (this.settleTimer !== undefined) {
      clearTimeout(this.settleTimer);
      this.settleTimer = undefined;
      this.refresh();
    } else if (this.viewMoved) {
      this.onMoveEnd();
    }
  }

  /** Nothing in flight: no model loading, no read, build, rebake or repaint waiting. */
  isIdle(): boolean {
    if (this.flushing || this.rafId !== undefined || this.paintRafId !== undefined || this.settleTimer !== undefined) return false;
    if (this.needsRead || this.needsBuild || this.pendingDemRegions.length) return false;
    for (const model of this.models.values()) if (model.state === 'loading') return false;
    return true;
  }

  /** Re-read the source and rebuild all instances (coalesced to one frame). */
  refresh() {
    this.schedule(true, true);
  }

  private schedule(read = false, build = false) {
    if (this.disposed) return;
    this.needsRead ||= read;
    this.needsBuild ||= build;
    if (this.rafId !== undefined) return;
    this.rafId = requestAnimationFrame(() => {
      this.rafId = undefined;
      this.flushing = true;
      void this.flush().finally(() => (this.flushing = false));
    });
  }

  private async flush() {
    if (this.disposed) return;
    if (this.needsRead || this.needsBuild) this.viewMoved = false; // read/built for the current view
    // Out of the zoom range: drop everything and don't even read the source.
    // MapLibre already skips render(), but the instances would still be built.
    if (this.hidden()) {
      this.needsRead = false;
      this.needsBuild = false;
      this.pendingDemRegions.length = 0;
      if (this.batches.size) {
        for (const [id, batch] of this.batches) this.disposeBatch(id, batch);
        this.map.triggerRepaint();
      }
      return;
    }
    if (this.needsRead) {
      this.needsRead = false;
      if (this.wholeDataset) await this.readGeoJson();
      else this.features = this.readTiles();
      if (this.disposed) return;
      this.needsBuild = true;
    }
    if (this.needsBuild) {
      this.needsBuild = false;
      this.pendingDemRegions.length = 0; // a build bakes fresh elevations anyway
      this.build();
      return;
    }
    const regions = this.pendingDemRegions;
    this.pendingDemRegions = [];
    if (regions.length === 0) return;
    let changed = 0;
    for (const batch of this.batches.values()) {
      if (!regions.some((r) => overlaps(r, batch.bounds))) continue;
      this.writeInstances(batch);
      changed++;
    }
    if (changed) this.map.triggerRepaint();
  }

  /** MapLibre semantics: hidden below `minzoom`, and at or above `maxzoom`. */
  private hidden(): boolean {
    const zoom = this.map.getZoom();
    return zoom < (this.config.minzoom ?? 0) || zoom >= (this.config.maxzoom ?? 24);
  }

  // ---------------------------------------------------------------- features

  /** Vector (and other tiled) sources: whatever is loaded for the viewport. */
  private readTiles(): RawFeature[] {
    // NB: tile pieces of one line/polygon all carry the same feature id, so
    // de-duplication cannot happen here — it happens per generated position.
    const out: RawFeature[] = [];
    let features: GeoJSONFeature[] = [];
    try {
      features = this.map.querySourceFeatures(this.config.sourceId, {
        sourceLayer: this.config.sourceLayer,
        filter: this.config.filter,
      }) as unknown as GeoJSONFeature[];
    } catch (err) {
      this.warnOnce('query', 'GltfLayer: querySourceFeatures failed', err);
      return out;
    }
    for (const f of features) {
      const bbox = bboxOf(f.geometry);
      if (!bbox) continue; // unsupported geometry
      out.push({ geometry: f.geometry, bbox, props: f.properties ?? {}, id: f.id });
    }
    if (features.length === 0 && this.map.isSourceLoaded(this.config.sourceId)) {
      this.warnOnce(
        'empty',
        `GltfLayer: source '${this.config.sourceId}' returned no features — a vector source only loads tiles while a style layer consumes it`,
      );
    }
    return out;
  }

  /** GeoJSON sources: the full dataset, filtered here. */
  private async readGeoJson(): Promise<void> {
    const src = this.map.getSource(this.config.sourceId) as
      | { getData?: () => Promise<unknown> }
      | undefined;
    if (!src?.getData) return;
    let data: unknown;
    try {
      data = await src.getData();
    } catch (err) {
      this.warnOnce('getdata', 'GltfLayer: could not read the geojson source', err);
      return;
    }
    if (this.disposed) return;

    const out: RawFeature[] = [];
    const globals = { zoom: this.map.getZoom(), globalState: globalStateOf(this.map) } as GlobalProperties;
    for (const f of featuresOf(data)) {
      const props = (f.properties ?? {}) as Record<string, unknown>;
      const bbox = bboxOf(f.geometry);
      if (!bbox) continue;
      // featureFilter wants an MVT-ish feature; 1/2/3 = point/line/polygon.
      if (!this.filter(globals, { type: mvtType(f.geometry), properties: props, id: f.id })) {
        continue;
      }
      out.push({ geometry: f.geometry, bbox, props, id: f.id });
    }
    this.features = out;
  }

  // ------------------------------------------------------------------- build

  /** Re-anchor the origin and rebuild every instance from the cached features. */
  private build() {
    const zoom = this.map.getZoom();
    const state = globalStateOf(this.map);
    const om = MercatorCoordinate.fromLngLat(this.map.getCenter(), 0);
    this.origin = { x: om.x, y: om.y, z: om.z, s: om.meterInMercatorCoordinateUnits() };

    let view = paddedBounds(this.map);
    // `model-thinning-distance`: full density up to it, halving every further
    // `thin` metres. Candidates are collected first and selected at the end, so
    // an over-budget view shrinks that distance instead of thinning everything.
    // The view is clipped to where density has fallen below ~1/64.
    const thin = this.config.thinningDistance(zoom, {}, undefined, state);
    const cam = Number.isFinite(thin) && thin > 0 ? cameraGround(this.map) : undefined;
    const maxMerc = cam ? thin * (1 + THIN_HALVINGS) * cam.s : Infinity;
    // `model-impostor-distance`: beyond it, draw the model's billboard instead.
    const impostorDistance = this.config.impostorDistance(zoom, {}, undefined, state);
    const impostorCam = Number.isFinite(impostorDistance) ? (cam ?? cameraGround(this.map)) : undefined;
    // Everything a top-down view of this zoom would show stays a full model, at
    // any pitch: a circle around the map centre through the (padded) screen
    // corners. Tilting moves the camera behind the centre, so the camera
    // distance alone would turn the far half of that area into billboards.
    const canvas = this.map.getCanvas();
    const topDownRadius =
      (Math.hypot(canvas.clientWidth, canvas.clientHeight) / 2) * (1 + FRAME_PAD) / (512 * 2 ** zoom);
    if (cam) {
      const nw = new MercatorCoordinate(cam.x - maxMerc, cam.y - maxMerc).toLngLat();
      const se = new MercatorCoordinate(cam.x + maxMerc, cam.y + maxMerc).toLngLat();
      view = {
        west: Math.max(view.west, nw.lng),
        east: Math.min(view.east, se.lng),
        north: Math.min(view.north, nw.lat),
        south: Math.max(view.south, se.lat),
      };
    }
    const clip: MercBounds = {
      minX: toMercX(view.west),
      minY: toMercY(view.south),
      maxX: toMercX(view.east),
      maxY: toMercY(view.north),
    };

    const visible = this.features.filter((f) => overlaps(view, f.bbox));
    // The budget is enforced on the collected on-screen candidates at the end;
    // the bbox-based pre-thinning only guards against runaway candidate counts.
    const max = this.config.maxInstances;
    const budget = max * CANDIDATE_FACTOR;
    const { cells, keep } = this.planCells(visible, zoom, state, view, budget);
    const inFrame = frameTest(this.map, FRAME_PAD);
    // Scan only the grid cells where the screen meets the ground.
    const rowSpan = frameGround(this.map, FRAME_PAD, cam, maxMerc);
    // Candidates whose survival scale reaches `cutoff` are vetoed before they
    // are built — most of a distant forest. The fitted scale never exceeds 1,
    // and it rarely moves much between builds, so start from the last one; if
    // that proves too tight (fewer candidates than the budget) the collection
    // reruns with 1. Either way the selected models are the same.
    let cutoff = cam ? Math.min(1, this.lastThinScale * THIN_HINT_SLACK) : 1;
    const accept: Accept | undefined = cam
      ? (px, py, id) =>
          survivalScale(Math.hypot(unitX(px) - cam.x, unitY(py) - cam.y) / cam.s, thin, id) < cutoff
      : undefined;

    // One position grid for the whole build: it resolves duplicates between
    // overlapping tile pieces AND between features that place on the same spot.
    const seen = new Set<number>();
    const limit = Math.round(this.config.maxInstances * HARD_STOP_FACTOR);
    let placed = 0;
    // Per candidate: `need` is the smallest factor on the thinning distance at
    // which it survives (0 without distance thinning); see survivalScale().
    const candidates: { f: RawFeature; point: PlacedPoint; x: number; y: number; need: number }[] = [];

    const groups = new globalThis.Map<string, { recs: Rec[]; colors?: Rgba[]; alphas?: number[] }>();
    const add = (f: RawFeature, point: PlacedPoint): boolean => {
      const x = point.x !== undefined ? unitX(point.x) : mercX(point.lng);
      const y = point.y !== undefined ? unitY(point.y) : mercY(point.lat);
      // A pitched, rotated view's lng/lat bbox is mostly off-screen — don't
      // spend the budget there.
      if (inFrame && !inFrame(x, y)) return true;
      const need = cam ? survivalScale(Math.hypot(x - cam.x, y - cam.y) / cam.s, thin, point.id) : 0;
      if (need >= cutoff) return true;
      candidates.push({ f, point, x, y, need });
      return candidates.length < budget;
    };
    const place = (f: RawFeature, point: PlacedPoint, x: number, y: number): boolean => {
      const id = point.id;
      const modelId = this.config.modelId(zoom, f.props, id, state);
      const model = this.models.get(modelId);
      if (!model) {
        this.warnOnce(
          `model:${modelId}`,
          `GltfLayer: no model '${modelId}' in \`models\` — feature skipped`,
        );
        return true;
      }
      if (model.state !== 'ready') return true; // rebuilds once it has loaded

      // Far away, the impostor (once baked) stands in for the model.
      const far =
        impostorCam !== undefined &&
        Math.hypot(x - impostorCam.x, y - impostorCam.y) / impostorCam.s >= impostorDistance &&
        Math.hypot(x - this.origin.x, y - this.origin.y) > topDownRadius;
      const key = far && this.models.get(impostorKey(modelId))?.state === 'ready' ? impostorKey(modelId) : modelId;

      let g = groups.get(key);
      if (!g) {
        groups.set(
          key,
          (g = {
            recs: [],
            colors: this.config.color ? [] : undefined,
            alphas: this.instanceOpacity ? [] : undefined,
          }),
        );
      }
      g.recs.push({
        lng: point.lng,
        lat: point.lat,
        east: (x - this.origin.x) / this.origin.s,
        north: (this.origin.y - y) / this.origin.s,
        rot: [...this.config.rotation(zoom, f.props, id, state)] as Vec3,
        scl: [...this.config.scale(zoom, f.props, id, state)] as Vec3,
        tr: [...this.config.translation(zoom, f.props, id, state)] as Vec3,
        props: f.props,
        id,
      });
      g.colors?.push(this.config.color!(zoom, f.props, id, state));
      g.alphas?.push(this.config.opacity!(zoom, f.props, id, state));
      return ++placed < limit;
    };

    const collect = () => {
      seen.clear();
      candidates.length = 0;
      outer: for (let i = 0; i < visible.length; i++) {
        const f = visible[i];
        const spacing = cells[i];
        switch (this.placementFor(f)) {
          case 'point':
            for (const [lng, lat] of pointsOf(f.geometry)) {
              // Points keep their SOURCE id, so `['id']` still addresses the OSM
              // feature; generated instances get a synthetic one instead.
              const key = positionKey(toMercX(lng), toMercY(lat));
              if (seen.has(key)) continue;
              seen.add(key);
              if (!add(f, { lng, lat, id: f.id as number })) break outer;
            }
            break;
          case 'line':
            if (!placeAlongLine(linesOf(f.geometry), spacing, clip, seen, (pt) => add(f, pt))) {
              break outer;
            }
            break;
          case 'fill':
            if (!scatterInPolygon(polysOf(f.geometry), spacing, clip, seen, (pt) => add(f, pt), keep, rowSpan, accept)) {
              break outer;
            }
            break;
          default:
            break;
        }
      }
    };
    collect();
    if (cutoff < 1 && candidates.length <= max) {
      cutoff = 1;
      collect();
    }

    if (candidates.length) {
      // Survival is decided by a hash of the position, so the models that remain
      // sit where they would at full density and never flicker between rebuilds.
      // Over budget: with distance thinning its distance shrinks (far models go
      // first), without it every candidate gets the same share.
      const scale = cam ? fitThinning(candidates, max) : 1;
      if (cam) this.lastThinScale = scale;
      const share = cam ? 1 : Math.min(1, max / candidates.length);
      if (share < 1) {
        this.warnOnce(
          'budget',
          `GltfLayer: ${candidates.length} instances in view exceed maxInstances (${max}) — ` +
            `keeping ${(share * 100).toFixed(0)}% of them; raise maxInstances, use a ` +
            `zoom-driven model-spacing or set model-thinning-distance`,
        );
      }
      for (const c of candidates) {
        if (cam ? c.need >= scale : survivalHash(c.point.id) >= share) continue;
        if (!place(c.f, c.point, c.x, c.y)) break;
      }
    }

    for (const [modelId, batch] of this.batches) {
      if (!groups.has(modelId)) this.disposeBatch(modelId, batch);
    }
    this.instanceFading = [...groups.values()].some((g) => g.alphas?.some((a) => a < 1));
    this.setFading(this.instanceFading || this.uniformFading);
    for (const [modelId, g] of groups) {
      const batch = this.batchFor(modelId, g.recs.length);
      batch.recs = g.recs;
      batch.bounds = recBounds(g.recs);
      this.writeInstances(batch, g.colors, g.alphas);
    }
    this.map.triggerRepaint();
  }

  /** Which placement mode a feature uses: the configured one, or by geometry. */
  private placementFor(f: RawFeature): Placement | undefined {
    const type = f.geometry.type;
    const wanted =
      this.config.placement !== 'auto'
        ? this.config.placement
        : type === 'Point' || type === 'MultiPoint'
          ? 'point'
          : type === 'LineString' || type === 'MultiLineString'
            ? 'line'
            : type === 'Polygon' || type === 'MultiPolygon'
              ? 'fill'
              : undefined;
    if (!wanted) return undefined;
    // An explicit placement that the geometry can't satisfy renders nothing.
    const ok =
      wanted === 'point'
        ? type === 'Point' || type === 'MultiPoint'
        : wanted === 'line'
          ? type === 'LineString' || type === 'MultiLineString'
          : type === 'Polygon' || type === 'MultiPolygon';
    if (!ok) {
      this.warnOnce(
        `placement:${type}`,
        `GltfLayer: model-placement '${wanted}' does not apply to ${type} — features skipped`,
      );
      return undefined;
    }
    return wanted;
  }

  /**
   * Spacing in real metres per visible feature, plus the fraction of
   * candidates to keep if the data would blow the instance budget.
   *
   * Thinning drops candidates and leaves the grid alone, so the models that
   * survive keep the positions they would have had at full density. Widening
   * the grid instead would move every single one whenever the viewport crossed
   * the budget — the thing this whole placement scheme exists to avoid.
   */
  private planCells(
    visible: RawFeature[],
    zoom: number,
    state: GlobalState,
    view: GeoBounds,
    max: number,
  ): { cells: number[]; keep: number } {
    const cells = new Array<number>(visible.length);
    let estimate = 0;

    for (let i = 0; i < visible.length; i++) {
      const f = visible[i];
      const mode = this.placementFor(f);
      if (mode === undefined) {
        cells[i] = 0;
        continue;
      }
      if (mode === 'point') {
        cells[i] = 0;
        estimate += pointsOf(f.geometry).length;
        continue;
      }
      const spacing = Math.max(MIN_SPACING_M, this.config.spacing(zoom, f.props, f.id, state));
      cells[i] = spacing;
      // Estimate only — the real grid is banded, this just sizes the budget.
      const cell = spacing * mercatorScale((f.bbox.south + f.bbox.north) / 2);
      estimate +=
        mode === 'fill'
          ? (polygonArea(polysOf(f.geometry)) / (cell * cell)) * inView(f.bbox, view)
          : lineLength(linesOf(f.geometry)) / cell;
    }

    if (estimate <= max) return { cells, keep: 1 };

    const keep = max / estimate;
    this.warnOnce(
      'candidates',
      `GltfLayer: ~${Math.round(estimate)} candidate instances around the view — ` +
        `pre-thinning to ${(keep * 100).toFixed(0)}%; use a zoom-driven model-spacing`,
    );
    return { cells, keep };
  }

  /** An existing batch if its buffers still fit, else a freshly sized one. */
  private batchFor(modelId: string, count: number): Batch {
    const existing = this.batches.get(modelId);
    // Reuse while the count fits and isn't so far below capacity that we'd be
    // uploading mostly-unused buffers — but never shrink below MIN_CAPACITY, or
    // a handful of features would reallocate on every single build.
    if (
      existing &&
      count <= existing.capacity &&
      (existing.capacity <= MIN_CAPACITY || count * 4 >= existing.capacity)
    ) {
      return existing;
    }
    if (existing) this.disposeBatch(modelId, existing);

    const model = this.models.get(modelId)!;
    const capacity = Math.max(MIN_CAPACITY, nextPow2(count));
    const meshes: THREE.InstancedMesh[] = [];
    let alphaAttr: THREE.InstancedBufferAttribute | undefined;
    if (this.instanceOpacity) {
      alphaAttr = new THREE.InstancedBufferAttribute(new Float32Array(capacity), 1);
    }
    for (const proto of model.protos) {
      const mesh = new THREE.InstancedMesh(proto.geometry, proto.material, capacity);
      mesh.frustumCulled = true;
      if (this.config.color) {
        mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3);
      }
      if (alphaAttr) proto.geometry.setAttribute('instanceAlpha', alphaAttr);
      meshes.push(mesh);
      this.group.add(mesh);
    }

    const batch: Batch = {
      meshes,
      protos: model.protos,
      capacity,
      recs: [],
      bounds: { west: 0, east: 0, south: 0, north: 0 },
      alphaAttr,
    };
    this.batches.set(modelId, batch);
    return batch;
  }

  /** (Re)compute a batch's instance matrices (and colours) against current terrain. */
  private writeInstances(batch: Batch, colors?: Rgba[], alphas?: number[]) {
    const n = batch.recs.length;
    const elevationAt = createElevationSampler(this.map);
    for (let i = 0; i < n; i++) {
      const rec = batch.recs[i];
      // Sampled per build rather than cached: point features move, so a cache
      // keyed by position would grow without bound.
      const ele = elevationAt(rec.lng, rec.lat);
      // Scene frame is (x=east, y=up, z=north); style vectors are [east, north, up].
      this.sPos.set(rec.east + rec.tr[0], ele + rec.tr[2], rec.north + rec.tr[1]);
      // rot = [around east, around north, around up]; order YXZ applies roll,
      // then pitch, then heading — so `rot[2]` is a compass heading (cw from north).
      this.sEuler.set(rec.rot[0] * D2R, rec.rot[2] * D2R, rec.rot[1] * D2R, 'YXZ');
      this.sQuat.setFromEuler(this.sEuler);
      this.sScale.set(rec.scl[0], rec.scl[1], rec.scl[2]);
      this.sBase.compose(this.sPos, this.sQuat, this.sScale);

      for (let m = 0; m < batch.meshes.length; m++) {
        this.sMat.multiplyMatrices(this.sBase, batch.protos[m].matrix);
        batch.meshes[m].setMatrixAt(i, this.sMat);
      }
    }

    for (const mesh of batch.meshes) {
      mesh.count = n;
      mesh.instanceMatrix.needsUpdate = true;
      mesh.computeBoundingSphere();
    }
    this.writePaint(batch, colors, alphas);
  }

  /** Write per-instance colours / alphas (either may be absent). */
  private writePaint(batch: Batch, colors?: Rgba[], alphas?: number[]) {
    const n = batch.recs.length;
    for (let i = 0; i < n; i++) {
      const color = colors?.[i];
      if (color) {
        this.sColor.setRGB(color[0], color[1], color[2], THREE.SRGBColorSpace);
        for (const mesh of batch.meshes) mesh.setColorAt(i, this.sColor);
      }
      const alpha = alphas?.[i];
      if (alpha !== undefined && batch.alphaAttr) batch.alphaAttr.setX(i, alpha);
    }
    for (const mesh of batch.meshes) {
      if (colors?.length && mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    }
    if (alphas?.length && batch.alphaAttr) batch.alphaAttr.needsUpdate = true;
  }

  /**
   * Re-evaluate per-instance colour/alpha after a global-state change, without
   * a rebuild (one per frame at most).
   */
  private schedulePaint() {
    if (this.disposed || this.paintRafId !== undefined) return;
    this.paintRafId = requestAnimationFrame(() => {
      this.paintRafId = undefined;
      const zoom = this.map.getZoom();
      const state: GlobalState = globalStateOf(this.map);
      const { color, opacity } = this.config;
      const instanceOpacity = this.instanceOpacity;
      let fading = false;
      for (const batch of this.batches.values()) {
        const colors = color && batch.recs.map((r) => color(zoom, r.props, r.id, state));
        const alphas = instanceOpacity ? batch.recs.map((r) => opacity!(zoom, r.props, r.id, state)) : undefined;
        fading ||= !!alphas?.some((a) => a < 1);
        this.writePaint(batch, colors, alphas);
      }
      this.instanceFading = fading;
      this.setFading(this.instanceFading || this.uniformFading);
      this.map.triggerRepaint();
    });
  }

  private disposeBatch(modelId: string, batch: Batch) {
    for (const mesh of batch.meshes) {
      this.group.remove(mesh);
      mesh.dispose(); // frees instanceMatrix/instanceColor, not the shared geometry
    }
    this.batches.delete(modelId);
  }

  // ------------------------------------------------------------------ models

  private loadModel(id: string, url: string) {
    const entry: ModelEntry = { url, state: 'loading', protos: [] };
    this.models.set(id, entry);
    this.loader.load(
      url,
      (gltf) => {
        if (this.disposed) return;
        entry.protos = this.flatten(gltf.scene);
        entry.state = 'ready';
        this.schedule(false, true); // features may already be waiting on this model
      },
      undefined,
      (err) => {
        entry.state = 'error';
        console.error(`GltfLayer: failed to load model '${id}' (${url})`, err);
      },
    );
  }

  /**
   * Flatten a loaded glTF scene into instanceable primitives: one entry per mesh,
   * carrying the mesh's transform *within* the file so a node hierarchy survives
   * instancing. Materials are patched for per-instance colour/alpha.
   */
  private flatten(root: THREE.Object3D): Proto[] {
    root.updateMatrixWorld(true);
    const protos: Proto[] = [];
    root.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (!mesh.isMesh) return;
      if ((mesh as unknown as { isSkinnedMesh?: boolean }).isSkinnedMesh) {
        this.warnOnce('skinned', 'GltfLayer: skinned meshes are not instanced — skipping one');
        return;
      }
      const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      for (const material of materials) this.patchMaterial(material);
      protos.push({
        geometry: mesh.geometry,
        material: mesh.material,
        matrix: mesh.matrixWorld.clone(),
      });
    });

    // Unless told otherwise: a closed, consistently wound mesh never shows a
    // back face, so drawing front faces only is free. A material that is also
    // used by an open mesh (leaf cards, ...) keeps both sides. "Closed" allows
    // a few stray open edges (an uncapped trunk base, a crown seam): real
    // models have them, while sheet geometry is open along most of its edges.
    if (this.config.doubleSided === undefined) {
      const open = new Set<THREE.Material>();
      const all = new Set<THREE.Material>();
      for (const proto of protos) {
        const closed = openEdgeShare(proto.geometry) <= MAX_OPEN_EDGE_SHARE;
        for (const m of Array.isArray(proto.material) ? proto.material : [proto.material]) {
          all.add(m);
          if (!closed) open.add(m);
        }
      }
      for (const m of all) if (m.side === THREE.DoubleSide && !open.has(m)) m.side = THREE.FrontSide;
    }
    return protos;
  }

  // ---------------------------------------------------------------- impostors

  /** Face the impostors toward a camera with this bearing (degrees, call each frame). */
  setViewBearing(bearing: number) {
    const b = (bearing * Math.PI) / 180;
    // Viewer looks along (sin b, cos b) in (east, north); its right is (cos b, -sin b).
    this.impostorRight.value.set(Math.cos(b), 0, -Math.sin(b));
  }

  /** Whether a loaded model still lacks its impostor (bake inside a render tick). */
  needsImpostors(): boolean {
    const impostorDistance = this.config.impostorDistance(this.map.getZoom(), {}, undefined, globalStateOf(this.map));
    if (!Number.isFinite(impostorDistance)) return false;
    for (const [id, model] of this.models) {
      if (model.state === 'ready' && !id.startsWith('\u0000') && !this.models.has(impostorKey(id))) return true;
    }
    return false;
  }

  /**
   * Render each loaded model once, from the side, into a small texture and
   * register a billboard quad showing it as the model's impostor. Uses the
   * renderer's live context — call from the layer's render(), then resetState().
   */
  bakeImpostors(renderer: THREE.WebGLRenderer, light: ImpostorLight) {
    for (const [id, model] of [...this.models]) {
      if (model.state !== 'ready' || id.startsWith('\u0000') || this.models.has(impostorKey(id))) continue;
      this.models.set(impostorKey(id), this.bakeImpostor(renderer, model, light));
    }
    this.schedule(false, true); // far instances can switch over now
  }

  private bakeImpostor(renderer: THREE.WebGLRenderer, model: ModelEntry, light: ImpostorLight): ModelEntry {
    // Bake frame: x=east, y=up, z=south — so a default camera (looking down -z)
    // looks north, with east to its right, like a map at bearing 0.
    const scene = new THREE.Scene();
    const group = new THREE.Group();
    group.scale.set(1, 1, -1);
    scene.add(group);

    const box = new THREE.Box3();
    const clones: THREE.Material[] = [];
    for (const proto of model.protos) {
      // Plain copies: the instancing patch isn't copied, and nothing fades here.
      const own = Array.isArray(proto.material) ? proto.material : [proto.material];
      const copies = own.map((m) => {
        const c = m.clone();
        if (c.defines) delete c.defines.MTK_INSTANCE_ALPHA;
        c.transparent = this.ownTransparent.get(m) ?? false;
        clones.push(c);
        return c;
      });
      const mesh = new THREE.Mesh(proto.geometry, Array.isArray(proto.material) ? copies : copies[0]);
      mesh.matrixAutoUpdate = false;
      mesh.matrix.copy(proto.matrix);
      group.add(mesh);
      proto.geometry.computeBoundingBox();
      box.union(proto.geometry.boundingBox!.clone().applyMatrix4(proto.matrix));
    }

    // Seen slightly from above, like a pitched map sees its distant models.
    // Square footprint: instances turn about the vertical axis, so the
    // silhouette can have any horizontal extent of the model.
    const center = box.getCenter(new THREE.Vector3());
    const half = Math.max(box.max.x - box.min.x, box.max.z - box.min.z) / 2 || 0.5;
    const e = (IMPOSTOR_ELEVATION * Math.PI) / 180;
    const radius = half * 2 + (box.max.y - box.min.y) + 10;
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, radius * 2);
    camera.position.set(center.x, center.y + Math.sin(e) * radius, Math.cos(e) * radius);
    camera.lookAt(center.x, center.y, 0);
    camera.updateMatrixWorld();
    // Frame the footprint cylinder (as a box) plus full height, in view space.
    const view = new THREE.Box3();
    const corner = new THREE.Vector3();
    for (const cx of [center.x - half, center.x + half]) {
      for (const cy of [box.min.y, box.max.y]) {
        for (const cz of [-half, half]) {
          view.expandByPoint(corner.set(cx, cy, cz).applyMatrix4(camera.matrixWorldInverse));
        }
      }
    }
    camera.left = view.min.x;
    camera.right = view.max.x;
    camera.bottom = view.min.y;
    camera.top = view.max.y;
    camera.updateProjectionMatrix();
    const width = view.max.x - view.min.x;
    const height = view.max.y - view.min.y;
    // Where the model's base (its origin) lands in the image, from the bottom.
    const baseY = new THREE.Vector3(center.x, 0, 0).applyMatrix4(camera.matrixWorldInverse).y - view.min.y;

    scene.add(new THREE.HemisphereLight(light.color, 0x9a9a9a, light.ambient));
    const sun = new THREE.DirectionalLight(light.color, light.directional);
    sun.position.copy(light.direction).normalize().multiplyScalar(1000);
    scene.add(sun, sun.target);
    scene.environment = light.environment;
    scene.environmentIntensity = light.environmentIntensity;

    const target = new THREE.WebGLRenderTarget(
      IMPOSTOR_RES,
      Math.max(1, Math.round((IMPOSTOR_RES * height) / width)),
      { samples: 4, colorSpace: THREE.LinearSRGBColorSpace },
    );
    target.texture.generateMipmaps = true;
    target.texture.minFilter = THREE.LinearMipmapLinearFilter;
    const clearColor = renderer.getClearColor(new THREE.Color());
    const clearAlpha = renderer.getClearAlpha();
    renderer.setRenderTarget(target);
    renderer.setClearColor(0x000000, 0);
    renderer.clear();
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    renderer.setClearColor(clearColor, clearAlpha);
    for (const c of clones) c.dispose();

    // The quad: image-sized, the model's origin on the ground, facing the viewer.
    const geometry = new THREE.PlaneGeometry(width, height);
    geometry.translate(center.x, height / 2 - baseY, 0);
    const material = new THREE.MeshBasicMaterial({ map: target.texture, alphaTest: 0.5 });
    material.alphaToCoverage = true; // soft edges with the map's MSAA
    material.onBeforeCompile = (shader) => {
      shader.uniforms.mtkRight = this.impostorRight;
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', IMPOSTOR_PARS_VERTEX)
        .replace('#include <project_vertex>', IMPOSTOR_PROJECT_VERTEX);
      shader.fragmentShader = shader.fragmentShader.replace(
        '#include <map_fragment>',
        IMPOSTOR_MAP_FRAGMENT,
      );
    };
    this.patchMaterial(material); // per-instance colour/alpha, on top of the billboard
    material.side = THREE.DoubleSide; // the quad turns; never cull it
    material.customProgramCacheKey = () => `mtk-impostor-${this.opacityKey}`;

    return {
      url: model.url,
      state: 'ready',
      protos: [{ geometry, material, matrix: new THREE.Matrix4() }],
      target,
    };
  }

  /**
   * Blend only while some instance is below full opacity (e.g. a zoom fade-in).
   * Opaque drawing lets the depth test skip hidden models' fragments, which in
   * a dense, low-angle forest is most of them.
   */
  private setFading(fading: boolean) {
    for (const model of this.models.values()) {
      for (const proto of model.protos) {
        const materials = Array.isArray(proto.material) ? proto.material : [proto.material];
        for (const m of materials) {
          const transparent = fading || (this.ownTransparent.get(m) ?? false);
          if (m.transparent === transparent) continue;
          m.transparent = transparent;
          m.needsUpdate = true;
        }
      }
    }
  }

  /** Make a stock three material honour `instanceColor` and our `instanceAlpha`. */
  private patchMaterial(material: THREE.Material) {
    if (this.patched.has(material)) return; // shared by several meshes
    this.patched.add(material);
    this.ownTransparent.set(material, material.transparent);
    if (this.config.doubleSided !== undefined) {
      material.side = this.config.doubleSided ? THREE.DoubleSide : THREE.FrontSide;
    }
    if (this.config.opacity) {
      material.defines = {
        ...material.defines,
        ...(this.instanceOpacity ? { MTK_INSTANCE_ALPHA: '' } : { MTK_OPACITY: '' }),
      };
      // Transparency is switched on per build, only while some instance is
      // actually below full opacity (see setFading) — blending costs a lot.
      // Solid objects with a uniform alpha read better with depth writes on —
      // without them every back face of the model shows through.
      material.depthWrite = true;
    }
    const previous = material.onBeforeCompile;
    material.onBeforeCompile = (shader, renderer) => {
      previous?.call(material, shader, renderer);
      if (this.uniformOpacity) shader.uniforms.mtkOpacity = this.opacityUniform;
      shader.vertexShader = shader.vertexShader
        .replace('#include <color_pars_vertex>', PARS_VERTEX)
        .replace('#include <color_vertex>', BODY_VERTEX);
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <color_pars_fragment>', PARS_FRAGMENT)
        .replace('#include <color_fragment>', BODY_FRAGMENT);
    };
    material.customProgramCacheKey = () => `mtk-gltf-${this.opacityKey}`;
    material.needsUpdate = true;
  }

  /** Shader variant for `model-opacity`: per-instance, uniform or none. */
  private get opacityKey(): string {
    return this.instanceOpacity ? 'a' : this.uniformOpacity ? 'u' : '';
  }

  private warnOnce(key: string, ...message: unknown[]) {
    if (this.warned.has(key)) return;
    this.warned.add(key);
    console.warn(...message);
  }

  /** Detach listeners and free all instance data (shared glTF geometry included). */
  dispose() {
    this.disposed = true;
    this.map.off('data', this.onData);
    this.map.off('moveend', this.onMoveEnd);
    this.map.off('move', this.onMove);
    if (this.rafId !== undefined) cancelAnimationFrame(this.rafId);
    if (this.paintRafId !== undefined) cancelAnimationFrame(this.paintRafId);
    for (const unwatch of this.unwatchGlobalState) unwatch();
    if (this.settleTimer !== undefined) clearTimeout(this.settleTimer);
    for (const [modelId, batch] of this.batches) this.disposeBatch(modelId, batch);
    for (const model of this.models.values()) {
      for (const proto of model.protos) {
        proto.geometry.dispose();
        for (const m of Array.isArray(proto.material) ? proto.material : [proto.material]) {
          m.dispose();
        }
      }
      model.target?.dispose();
    }
    this.models.clear();
    this.scene.remove(this.group);
  }
}

// --------------------------------------------------------------------- utils

/** Minimal structural types — avoids a hard dependency on @types/geojson. */
interface GeoJSONGeometry {
  type: string;
  coordinates?: unknown;
}
interface GeoJSONFeature {
  id?: string | number;
  properties?: Record<string, unknown> | null;
  geometry: GeoJSONGeometry;
}

/** Coordinates normalised to multi-form, so callers handle one shape. */
function pointsOf(g: GeoJSONGeometry): number[][] {
  return g.type === 'Point' ? [g.coordinates as number[]] : (g.coordinates as number[][]);
}
function linesOf(g: GeoJSONGeometry): number[][][] {
  return g.type === 'LineString' ? [g.coordinates as number[][]] : (g.coordinates as number[][][]);
}
function polysOf(g: GeoJSONGeometry): number[][][][] {
  return g.type === 'Polygon'
    ? [g.coordinates as number[][][]]
    : (g.coordinates as number[][][][]);
}

/** MVT feature type, as `featureFilter` and `['geometry-type']` expect it. */
function mvtType(g: GeoJSONGeometry | undefined): 1 | 2 | 3 {
  if (!g) return 1;
  if (g.type === 'LineString' || g.type === 'MultiLineString') return 2;
  if (g.type === 'Polygon' || g.type === 'MultiPolygon') return 3;
  return 1;
}

/** Geographic bounds of any supported geometry; undefined if unsupported. */
function bboxOf(g: GeoJSONGeometry | undefined): GeoBounds | undefined {
  if (!g) return undefined;
  const b = { west: Infinity, east: -Infinity, south: Infinity, north: -Infinity };
  let seen = false;
  const walk = (node: unknown): void => {
    const arr = node as unknown[];
    if (!Array.isArray(arr) || arr.length === 0) return;
    if (typeof arr[0] === 'number') {
      const lng = arr[0] as number;
      const lat = arr[1] as number;
      if (lng < b.west) b.west = lng;
      if (lng > b.east) b.east = lng;
      if (lat < b.south) b.south = lat;
      if (lat > b.north) b.north = lat;
      seen = true;
      return;
    }
    for (const child of arr) walk(child);
  };
  switch (g.type) {
    case 'Point':
    case 'MultiPoint':
    case 'LineString':
    case 'MultiLineString':
    case 'Polygon':
    case 'MultiPolygon':
      walk(g.coordinates);
      break;
    default:
      return undefined;
  }
  return seen ? b : undefined;
}

function featuresOf(data: unknown): GeoJSONFeature[] {
  const d = data as { type?: string; features?: GeoJSONFeature[] } & GeoJSONFeature;
  if (d?.type === 'FeatureCollection') return d.features ?? [];
  if (d?.type === 'Feature') return [d];
  return [];
}

function nextPow2(n: number): number {
  let c = 1;
  while (c < n) c *= 2;
  return c;
}

function toOrigin(lngLat: LngLat): SceneOrigin {
  const om = MercatorCoordinate.fromLngLat(lngLat, 0);
  return { x: om.x, y: om.y, z: om.z, s: om.meterInMercatorCoordinateUnits() };
}

/** The visible bounds, grown by {@link CULL_PAD} so models don't pop at the edge. */
function paddedBounds(map: Map): GeoBounds {
  const b = map.getBounds();
  const dLng = (b.getEast() - b.getWest()) * CULL_PAD;
  const dLat = (b.getNorth() - b.getSouth()) * CULL_PAD;
  return {
    west: b.getWest() - dLng,
    east: b.getEast() + dLng,
    south: b.getSouth() - dLat,
    north: b.getNorth() + dLat,
  };
}

/** Stable pseudo-random 0..1 per instance: decides which ones a thinning keeps. */
function survivalHash(id: number): number {
  return hash32(id | 0, 0x51ed) / 4294967296;
}

/**
 * Distance thinning keeps a model at `d` metres with probability 1 up to
 * `full`, then 2^-((d - full) / full). With a stable u = survivalHash(id) that
 * is: kept iff u < 2^(1 - d / full), i.e. iff the full-density distance is at
 * least d / (1 - log2 u). Returned as a factor on `full`: the model survives
 * every scale above this value.
 */
function survivalScale(d: number, full: number, id: number): number {
  return d / (full * (1 - Math.log2(survivalHash(id))));
}

/**
 * Largest factor (<= 1) on the full-density distance that keeps the survivors
 * within `max` — so an over-budget view loses its far models first. Exact: the
 * max-th smallest survival scale.
 */
function fitThinning(cands: { need: number }[], max: number): number {
  let within = 0;
  for (const c of cands) if (c.need < 1) within++;
  if (within <= max) return 1;
  const needs = new Float64Array(cands.length);
  for (let i = 0; i < cands.length; i++) needs[i] = cands[i].need;
  needs.sort();
  return needs[max];
}

/** Web-mercator x (0..1) of a longitude — same as MercatorCoordinate.fromLngLat. */
function mercX(lng: number): number {
  return (180 + lng) / 360;
}

/** Web-mercator y (0..1) of a latitude — same as MercatorCoordinate.fromLngLat. */
function mercY(lat: number): number {
  return (180 - (180 / Math.PI) * Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360))) / 360;
}

/**
 * Is a mercator point within the screen, grown by `pad` of its size at the
 * sides and bottom (not the top, see FRAME_PAD)?
 * Projects onto flat ground at the centre's elevation (terrain relief only
 * shifts points a little, which the padding absorbs), with MapLibre's
 * (internal) pixel matrix; undefined when that isn't available.
 */
function frameTest(map: Map, pad: number): ((x: number, y: number) => boolean) | undefined {
  type T = { worldSize: number; width: number; height: number; elevation?: number;
    _pixelMatrix?: ArrayLike<number>; _pixelMatrix3D?: ArrayLike<number> };
  const m = map as unknown as { transform?: T; _camera?: { transform?: T } };
  const t = m.transform ?? m._camera?.transform;
  const M = map.terrain ? t?._pixelMatrix3D : t?._pixelMatrix;
  if (!t || !M || !t.worldSize) return undefined;
  const z = map.terrain ? (t.elevation ?? 0) : 0;
  const ws = t.worldSize;
  const px = t.width * pad;
  const py = t.height * pad;
  return (x, y) => {
    const X = x * ws;
    const Y = y * ws;
    const w = M[3] * X + M[7] * Y + M[11] * z + M[15];
    if (w <= 0) return false; // behind the camera
    const sx = (M[0] * X + M[4] * Y + M[8] * z + M[12]) / w;
    const sy = (M[1] * X + M[5] * Y + M[9] * z + M[13]) / w;
    return sx >= -px && sx <= t.width + px && sy >= 0 && sy <= t.height + py;
  };
}

/**
 * Where the screen meets flat ground at the centre's elevation, as a row
 * pre-filter for grid placement: the screen corners (margin as in frameTest)
 * are cast as rays onto that plane. Rays above the horizon, and ground more
 * than `maxUnits` (mercator units) from the camera, end at that distance.
 * Undefined when MapLibre's internals are missing, or the view reaches the
 * horizon with no distance limit.
 */
function frameGround(
  map: Map,
  pad: number,
  cam: { x: number; y: number } | undefined,
  maxUnits: number,
): RowSpan | undefined {
  type T = { worldSize: number; width: number; height: number; elevation?: number;
    _pixelMatrix?: ArrayLike<number>; _pixelMatrix3D?: ArrayLike<number> };
  const m = map as unknown as { transform?: T; _camera?: { transform?: T } };
  const t = m.transform ?? m._camera?.transform;
  const M = map.terrain ? t?._pixelMatrix3D : t?._pixelMatrix;
  if (!t || !M || !t.worldSize) return undefined;
  const inv = new THREE.Matrix4().fromArray(Array.from(M)).invert();
  const z = map.terrain ? (t.elevation ?? 0) : 0;
  const ws = t.worldSize;
  const a = new THREE.Vector4();
  const b = new THREE.Vector4();

  // Ground point (mercator units) under screen pixel (sx, sy), or undefined.
  const cast = (sx: number, sy: number): [number, number] | undefined => {
    a.set(sx, sy, 0, 1).applyMatrix4(inv).divideScalar(a.w);
    b.set(sx, sy, 1, 1).applyMatrix4(inv).divideScalar(b.w);
    const dirX = (b.x - a.x) / ws;
    const dirY = (b.y - a.y) / ws;
    const k = b.z !== a.z ? (z - a.z) / (b.z - a.z) : NaN;
    const gx = (a.x + (b.x - a.x) * k) / ws;
    const gy = (a.y + (b.y - a.y) * k) / ws;
    const hit = Number.isFinite(k) && M[3] * gx * ws + M[7] * gy * ws + M[11] * z + M[15] > 0;
    if (!cam) return hit ? [gx, gy] : undefined;
    // Limit to maxUnits from the camera — along the ray's heading if it misses.
    let dx = hit ? gx - cam.x : dirX;
    let dy = hit ? gy - cam.y : dirY;
    const len = Math.hypot(dx, dy);
    if (!(len > 0)) return undefined;
    if (!hit || len > maxUnits) {
      dx *= maxUnits / len;
      dy *= maxUnits / len;
    }
    return [cam.x + dx, cam.y + dy];
  };

  const px = t.width * pad;
  const py = t.height * pad;
  const corners = [
    cast(-px, t.height + py),
    cast(t.width + px, t.height + py),
    cast(t.width + px, 0),
    cast(-px, 0),
  ];
  if (corners.some((c) => !c)) return undefined;
  const poly = (corners as [number, number][]).map(([x, y]) => [fromUnitX(x), fromUnitY(y)]);

  // x-range of the (convex) quad within the slab y0..y1: its edges clipped to it.
  return (y0, y1) => {
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < poly.length; i++) {
      const [ax, ay] = poly[i];
      const [bx, by] = poly[(i + 1) % poly.length];
      let t0 = 0;
      let t1 = 1;
      const dy = by - ay;
      if (dy === 0) {
        if (ay < y0 || ay > y1) continue;
      } else {
        let ta = (y0 - ay) / dy;
        let tb = (y1 - ay) / dy;
        if (ta > tb) [ta, tb] = [tb, ta];
        t0 = Math.max(0, ta);
        t1 = Math.min(1, tb);
        if (t0 > t1) continue;
      }
      const xa = ax + (bx - ax) * t0;
      const xb = ax + (bx - ax) * t1;
      lo = Math.min(lo, xa, xb);
      hi = Math.max(hi, xa, xb);
    }
    return lo <= hi ? [lo, hi] : null;
  };
}

/**
 * Share (0..1) of a mesh's edges that are not closed and consistently wound —
 * i.e. not used once in each direction. 0 for a watertight shell, near 1 for
 * loose sheets. Positions are welded first, since glTF splits vertices at UV
 * and normal seams.
 */
function openEdgeShare(geometry: THREE.BufferGeometry): number {
  const pos = geometry.getAttribute('position');
  if (!pos) return 1;
  const index = geometry.getIndex();
  const n = index ? index.count : pos.count;
  if (n < 3 || n % 3 !== 0) return 1;

  const welded = new Int32Array(pos.count);
  const ids = new globalThis.Map<string, number>();
  for (let i = 0; i < pos.count; i++) {
    const key = `${Math.round(pos.getX(i) * 1e4)},${Math.round(pos.getY(i) * 1e4)},${Math.round(pos.getZ(i) * 1e4)}`;
    let id = ids.get(key);
    if (id === undefined) ids.set(key, (id = ids.size));
    welded[i] = id;
  }

  // Per undirected edge: +1 for each use as low→high, −1 as high→low.
  const stride = ids.size;
  const balance = new globalThis.Map<number, number>();
  const vertex = (i: number) => welded[index ? index.getX(i) : i];
  for (let t = 0; t < n; t += 3) {
    const tri = [vertex(t), vertex(t + 1), vertex(t + 2)];
    if (tri[0] === tri[1] || tri[1] === tri[2] || tri[0] === tri[2]) continue; // degenerate
    for (let k = 0; k < 3; k++) {
      const u = tri[k];
      const v = tri[(k + 1) % 3];
      const key = u < v ? u * stride + v : v * stride + u;
      balance.set(key, (balance.get(key) ?? 0) + (u < v ? 1 : -1));
    }
  }
  if (balance.size === 0) return 1;
  let open = 0;
  for (const b of balance.values()) if (b !== 0) open++;
  return open / balance.size;
}

/** Fraction of a feature's bbox that the (clipped, padded) view covers (0..1). */
function inView(bbox: GeoBounds, view: GeoBounds): number {
  const w = Math.min(bbox.east, view.east) - Math.max(bbox.west, view.west);
  const h = Math.min(bbox.north, view.north) - Math.max(bbox.south, view.south);
  if (w <= 0 || h <= 0) return 0;
  const full = (bbox.east - bbox.west) * (bbox.north - bbox.south);
  return full > 0 ? Math.min(1, (w * h) / full) : 1;
}

/**
 * The camera's ground position in mercator units, plus mercator units per metre
 * there. Reads MapLibre's (internal) transform; falls back to the map centre.
 */
function cameraGround(map: Map): { x: number; y: number; s: number } {
  type T = { getCameraLngLat?: () => LngLat };
  const m = map as unknown as { transform?: T; _camera?: { transform?: T } };
  const lngLat = (m.transform ?? m._camera?.transform)?.getCameraLngLat?.() ?? map.getCenter();
  const c = MercatorCoordinate.fromLngLat(lngLat, 0);
  return { x: c.x, y: c.y, s: c.meterInMercatorCoordinateUnits() };
}

function recBounds(recs: Rec[]): GeoBounds {
  const b = { west: 180, east: -180, south: 90, north: -90 };
  for (const r of recs) {
    if (r.lng < b.west) b.west = r.lng;
    if (r.lng > b.east) b.east = r.lng;
    if (r.lat < b.south) b.south = r.lat;
    if (r.lat > b.north) b.north = r.lat;
  }
  return b;
}

/** Do two geographic bounding boxes intersect? (edge-touching counts.) */
function overlaps(a: GeoBounds, b: GeoBounds): boolean {
  return a.west <= b.east && a.east >= b.west && a.south <= b.north && a.north >= b.south;
}
