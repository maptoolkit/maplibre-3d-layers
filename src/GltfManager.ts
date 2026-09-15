import { LngLat, MercatorCoordinate } from 'maplibre-gl';
import type { LngLatLike, Map } from 'maplibre-gl';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { featureFilter } from '@maplibre/maplibre-gl-style-spec';
import type { FilterSpecification, GlobalProperties } from '@maplibre/maplibre-gl-style-spec';
import { tileBounds } from './ElevationCache';
import type { GeoBounds } from './ElevationCache';
import type { Evaluator, Rgba, Vec3 } from './expression';
import type { Placement } from './GltfLayer';
import {
  lineLength,
  mercatorScale,
  placeAlongLine,
  polygonArea,
  positionKey,
  scatterInPolygon,
  toMercX,
  toMercY,
} from './placement';
import type { MercBounds, PlacedPoint } from './placement';

const D2R = Math.PI / 180;

/** Viewport padding (fraction of the view size) when culling source features. */
const CULL_PAD = 0.25;

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
  /** Upper bound on instances per build; denser data is thinned to fit. */
  maxInstances: number;
  /** Layer is hidden below this zoom (MapLibre semantics). */
  minzoom?: number;
  /** Layer is hidden at and above this zoom (MapLibre semantics). */
  maxzoom?: number;
}

/** Smallest spacing we honour — also keeps the placement grid keys collision-free. */
const MIN_SPACING_M = 1;

/** Safety net if the pre-pass estimate was too optimistic. */
const HARD_STOP_FACTOR = 1.2;

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
}

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
`;

const BODY_FRAGMENT = /* glsl */ `
#include <color_fragment>
#if defined( USE_INSTANCING_COLOR ) && !defined( USE_COLOR ) && !defined( USE_COLOR_ALPHA )
  diffuseColor.rgb *= vColor.rgb;
#endif
#ifdef MTK_INSTANCE_ALPHA
  diffuseColor.a *= vInstanceAlpha;
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

  /** Cached source features; re-read on source data, re-culled on move. */
  private features: RawFeature[] = [];
  /** geojson sources deliver the whole dataset at once — vector ones per tile. */
  private readonly wholeDataset: boolean;

  private readonly onData: (ev: unknown) => void;
  private readonly onMoveEnd: () => void;
  private needsRead = false;
  private needsBuild = false;
  private pendingDemRegions: GeoBounds[] = [];
  private rafId?: number;
  private disposed = false;

  // Scratch objects reused while writing instance matrices.
  private readonly sPos = new THREE.Vector3();
  private readonly sQuat = new THREE.Quaternion();
  private readonly sEuler = new THREE.Euler();
  private readonly sScale = new THREE.Vector3();
  private readonly sBase = new THREE.Matrix4();
  private readonly sMat = new THREE.Matrix4();
  private readonly sColor = new THREE.Color();
  /** Mutable LngLatLike reused for terrain/mercator lookups. */
  private readonly sPoint = { lng: 0, lat: 0 } as LngLatLike & { lng: number; lat: number };

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
        this.refresh();
      }
    };
    // A whole-dataset source only needs re-culling when the view moves; a tiled
    // one has to be re-queried, because its result IS the viewport.
    this.onMoveEnd = () => (this.wholeDataset ? this.schedule(false, true) : this.refresh());
    map.on('data', this.onData);
    map.on('moveend', this.onMoveEnd);

    this.refresh();
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
      void this.flush();
    });
  }

  private async flush() {
    if (this.disposed) return;
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
    const globals = { zoom: this.map.getZoom() } as GlobalProperties;
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
    const om = MercatorCoordinate.fromLngLat(this.map.getCenter(), 0);
    this.origin = { x: om.x, y: om.y, z: om.z, s: om.meterInMercatorCoordinateUnits() };

    const view = paddedBounds(this.map);
    const clip: MercBounds = {
      minX: toMercX(view.west),
      minY: toMercY(view.south),
      maxX: toMercX(view.east),
      maxY: toMercY(view.north),
    };

    const visible = this.features.filter((f) => overlaps(view, f.bbox));
    const { cells, keep } = this.planCells(visible, zoom);

    // One position grid for the whole build: it resolves duplicates between
    // overlapping tile pieces AND between features that place on the same spot.
    const seen = new Set<number>();
    const limit = Math.round(this.config.maxInstances * HARD_STOP_FACTOR);
    let placed = 0;

    const groups = new globalThis.Map<string, { recs: Rec[]; colors?: Rgba[]; alphas?: number[] }>();
    const add = (f: RawFeature, point: PlacedPoint): boolean => {
      const id = point.id;
      const modelId = this.config.modelId(zoom, f.props, id);
      const model = this.models.get(modelId);
      if (!model) {
        this.warnOnce(
          `model:${modelId}`,
          `GltfLayer: no model '${modelId}' in \`models\` — feature skipped`,
        );
        return true;
      }
      if (model.state !== 'ready') return true; // rebuilds once it has loaded

      let g = groups.get(modelId);
      if (!g) {
        groups.set(
          modelId,
          (g = {
            recs: [],
            colors: this.config.color ? [] : undefined,
            alphas: this.config.opacity ? [] : undefined,
          }),
        );
      }
      const fm = MercatorCoordinate.fromLngLat(this.point(point.lng, point.lat), 0);
      g.recs.push({
        lng: point.lng,
        lat: point.lat,
        east: (fm.x - this.origin.x) / this.origin.s,
        north: (this.origin.y - fm.y) / this.origin.s,
        rot: [...this.config.rotation(zoom, f.props, id)] as Vec3,
        scl: [...this.config.scale(zoom, f.props, id)] as Vec3,
        tr: [...this.config.translation(zoom, f.props, id)] as Vec3,
      });
      g.colors?.push(this.config.color!(zoom, f.props, id));
      g.alphas?.push(this.config.opacity!(zoom, f.props, id));
      return ++placed < limit;
    };

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
          if (!scatterInPolygon(polysOf(f.geometry), spacing, clip, seen, (pt) => add(f, pt), keep)) {
            break outer;
          }
          break;
        default:
          break;
      }
    }

    for (const [modelId, batch] of this.batches) {
      if (!groups.has(modelId)) this.disposeBatch(modelId, batch);
    }
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
      const spacing = Math.max(MIN_SPACING_M, this.config.spacing(zoom, f.props, f.id));
      cells[i] = spacing;
      // Estimate only — the real grid is banded, this just sizes the budget.
      const cell = spacing * mercatorScale((f.bbox.south + f.bbox.north) / 2);
      estimate +=
        mode === 'fill'
          ? (polygonArea(polysOf(f.geometry)) / (cell * cell)) * this.inView(f.bbox)
          : lineLength(linesOf(f.geometry)) / cell;
    }

    const max = this.config.maxInstances;
    if (estimate <= max) return { cells, keep: 1 };

    const keep = max / estimate;
    this.warnOnce(
      'budget',
      `GltfLayer: ~${Math.round(estimate)} instances exceeds maxInstances (${max}) — ` +
        `keeping ${(keep * 100).toFixed(0)}% of them; raise maxInstances or use a ` +
        `zoom-driven model-spacing`,
    );
    return { cells, keep };
  }

  /** Fraction of a feature's bbox that the padded viewport covers (0..1). */
  private inView(bbox: GeoBounds): number {
    const view = paddedBounds(this.map);
    const w = Math.min(bbox.east, view.east) - Math.max(bbox.west, view.west);
    const h = Math.min(bbox.north, view.north) - Math.max(bbox.south, view.south);
    if (w <= 0 || h <= 0) return 0;
    const full = (bbox.east - bbox.west) * (bbox.north - bbox.south);
    return full > 0 ? Math.min(1, (w * h) / full) : 1;
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
    if (this.config.opacity) {
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
    for (let i = 0; i < n; i++) {
      const rec = batch.recs[i];
      // Sampled per build rather than cached: point features move, so a cache
      // keyed by position would grow without bound.
      const ele = this.map.queryTerrainElevation(this.point(rec.lng, rec.lat)) ?? 0;
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
      const color = colors?.[i];
      if (color) {
        this.sColor.setRGB(color[0], color[1], color[2], THREE.SRGBColorSpace);
        for (const mesh of batch.meshes) mesh.setColorAt(i, this.sColor);
      }
      const alpha = alphas?.[i];
      if (alpha !== undefined && batch.alphaAttr) batch.alphaAttr.setX(i, alpha);
    }

    for (const mesh of batch.meshes) {
      mesh.count = n;
      mesh.instanceMatrix.needsUpdate = true;
      if (colors?.length && mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      mesh.computeBoundingSphere();
    }
    if (alphas?.length && batch.alphaAttr) batch.alphaAttr.needsUpdate = true;
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
    return protos;
  }

  /** Make a stock three material honour `instanceColor` and our `instanceAlpha`. */
  private patchMaterial(material: THREE.Material) {
    if (this.patched.has(material)) return; // shared by several meshes
    this.patched.add(material);
    if (this.config.opacity) {
      material.defines = { ...material.defines, MTK_INSTANCE_ALPHA: '' };
      material.transparent = true;
      // Solid objects with a uniform alpha read better with depth writes on —
      // without them every back face of the model shows through.
      material.depthWrite = true;
    }
    const previous = material.onBeforeCompile;
    material.onBeforeCompile = (shader, renderer) => {
      previous?.call(material, shader, renderer);
      shader.vertexShader = shader.vertexShader
        .replace('#include <color_pars_vertex>', PARS_VERTEX)
        .replace('#include <color_vertex>', BODY_VERTEX);
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <color_pars_fragment>', PARS_FRAGMENT)
        .replace('#include <color_fragment>', BODY_FRAGMENT);
    };
    material.customProgramCacheKey = () => `mtk-gltf-${this.config.opacity ? 'a' : ''}`;
    material.needsUpdate = true;
  }

  private point(lng: number, lat: number): LngLatLike {
    this.sPoint.lng = lng;
    this.sPoint.lat = lat;
    return this.sPoint;
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
    if (this.rafId !== undefined) cancelAnimationFrame(this.rafId);
    for (const [modelId, batch] of this.batches) this.disposeBatch(modelId, batch);
    for (const model of this.models.values()) {
      for (const proto of model.protos) {
        proto.geometry.dispose();
        for (const m of Array.isArray(proto.material) ? proto.material : [proto.material]) {
          m.dispose();
        }
      }
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
