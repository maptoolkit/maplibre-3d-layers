import { MercatorCoordinate, LngLat } from 'maplibre-gl';
import type { CoveringTilesOptions, Map } from 'maplibre-gl';
import * as THREE from 'three';
import type { ElevationCache } from './ElevationCache';
import { createMercatorElevationSampler, tileBounds } from './ElevationCache';
import type { GeoBounds } from './ElevationCache';
import type { TileRequest, TileResult, FeatureSpan } from './mvt3dGeometry';
import type { ColorEval, OpacityEval } from './paint';
import type { FilterSpecification } from '@maplibre/maplibre-gl-style-spec';
// Inlined as a base64 blob so the built bundle stays single-file — no separate
// worker asset to host. Works for both the ESM package and the <script> build.
import Mvt3dWorker from './mvt3dWorker.ts?worker&inline';

/** Vertices this close (metres) above a building's base count as its base. */
const BASE_EPSILON_M = 0.05;

/** Re-anchor the scene origin once the map centre drifts this far from it (metres). */
const REBASE_DISTANCE_M = 3000;

// Replicates MapLibre's fill-extrusion vertex lighting (a flat directional ramp,
// NOT PBR) so the 3D meshes shade like the native fill-extrusion LOD1. Works in
// sRGB and outputs directly (no colour-management conversion), like MapLibre.
// `normal`/`color` are our per-vertex attributes in the (east, up, north) frame;
// u_lightpos is the style light mapped into that same frame (see syncLight()).
const FILL3D_VERTEX = /* glsl */ `
  attribute vec4 color;
  uniform vec3 u_lightpos;
  uniform float u_lightintensity;
  uniform vec3 u_lightcolor;
  varying vec4 v_color;
  void main() {
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
    v_color = vec4(lit, color.a);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const FILL3D_FRAGMENT = /* glsl */ `
  precision mediump float;
  varying vec4 v_color;
  void main() { gl_FragColor = v_color; }
`;

// Scratch colours reused by syncLight() (called every frame).
const SCRATCH_COLOR = new THREE.Color();
const SCRATCH_SRGB = new THREE.Color();

function createFill3DMaterial(transparent: boolean): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      u_lightpos: { value: new THREE.Vector3(0.288, 0.996, 0.498) },
      u_lightintensity: { value: 0.5 },
      u_lightcolor: { value: new THREE.Vector3(1, 1, 1) },
    },
    vertexShader: FILL3D_VERTEX,
    fragmentShader: FILL3D_FRAGMENT,
    // Backface culling (like fill-extrusion): only camera-facing outer faces
    // draw, so semi-transparent buildings don't show their inner/back walls.
    // BackSide is the outer shell here (source winding + scale(1,1,-1) reflection).
    side: THREE.BackSide,
    transparent,
  });
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
  /** Worker positions with y = height above the feature base (no terrain yet). */
  positionBase?: Float32Array;
  features?: FeatureSpan[];
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
 * bakes terrain elevation per feature centroid into the vertex Y, and re-bakes
 * whenever new DEM tiles stream in (via the shared {@link ElevationCache}).
 */
export class Fill3DManager {
  /** Current scene anchor; read by the layer to build the projection matrix. */
  origin: SceneOrigin;

  private readonly map: Map;
  private readonly scene: THREE.Scene;
  private readonly group: THREE.Group;
  private readonly elevation: ElevationCache;
  private readonly terrainSourceId: string;
  private readonly worker: Worker;
  private readonly tiles = new globalThis.Map<string, TileEntry>();
  private readonly material: THREE.ShaderMaterial;
  private readonly config: Fill3DConfig;
  private readonly onData: (ev: unknown) => void;
  private readonly onMoveEnd: () => void;
  private onSourceData?: (ev: unknown) => void;

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

    // Shades like MapLibre's fill-extrusion (flat directional ramp). Per-vertex
    // sRGB colour (fill-3d-color) + alpha (fill-3d-opacity) via the `color` attr.
    this.material = createFill3DMaterial(config.transparent);

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
    map.on('data', this.onData);
    map.on('moveend', this.onMoveEnd);

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
      if (entry.mesh) this.group.remove(entry.mesh);
      entry.geometry?.dispose();
      this.tiles.delete(key);
    }
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

    fetch(url)
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
        };
        this.worker.postMessage(req, [buffer]);
      })
      .catch(() => {
        const entry = this.tiles.get(key);
        if (entry) entry.state = 'empty';
      });
  }

  private onTileParsed(result: TileResult) {
    const entry = this.tiles.get(result.key);
    if (!entry || entry.state !== 'loading') return; // evicted mid-flight

    if (result.index.length === 0) {
      entry.state = 'empty';
      return;
    }

    const geometry = new THREE.BufferGeometry();
    // Copy positions so we can keep the base-relative Y for re-baking.
    geometry.setAttribute('position', new THREE.BufferAttribute(result.position.slice(), 3));
    geometry.setAttribute('normal', new THREE.BufferAttribute(result.normal, 3));
    geometry.setAttribute('color', this.colorAttribute(result));
    geometry.setIndex(new THREE.BufferAttribute(result.index, 1));

    const mesh = new THREE.Mesh(geometry, this.material);
    mesh.frustumCulled = true;
    this.group.add(mesh);

    entry.state = 'ready';
    entry.tileScale = result.tileScale;
    entry.mesh = mesh;
    entry.geometry = geometry;
    entry.positionBase = result.position; // y = base-relative height
    entry.features = result.features;

    this.placeMesh(entry);
    this.bake(entry);
    this.map.triggerRepaint();
  }

  /** Evaluate fill-3d-color / fill-3d-opacity per feature into a per-vertex RGBA attribute.
   *  Colours stay sRGB — the shader shades and outputs in sRGB, matching fill-extrusion. */
  private colorAttribute(result: TileResult): THREE.BufferAttribute {
    const zoom = this.map.getZoom();
    const rgba = new Float32Array((result.position.length / 3) * 4);
    for (const b of result.features) {
      const [r, g, bl, ca] = this.config.color(zoom, b.properties);
      const a = ca * this.config.opacity(zoom, b.properties);
      const end = b.start + b.count;
      for (let v = b.start; v < end; v++) {
        rgba[v * 4] = r;
        rgba[v * 4 + 1] = g;
        rgba[v * 4 + 2] = bl;
        rgba[v * 4 + 3] = a;
      }
    }
    return new THREE.BufferAttribute(rgba, 4);
  }

  /**
   * Switch the shared material between the two rendering passes. A depth-only
   * pre-pass establishes the nearest surface per pixel; the colour pass then only
   * blends fragments at that depth — so interior/back walls of semi-transparent
   * buildings are hidden (like fill-extrusion's opaque-then-composite), while the
   * frontmost outer shell keeps its smooth per-feature alpha.
   */
  setPass(pass: 'depth' | 'color') {
    const m = this.material;
    m.depthTest = true;
    m.depthFunc = THREE.LessEqualDepth;
    if (pass === 'depth') {
      m.colorWrite = false;
      m.depthWrite = true;
    } else {
      m.colorWrite = true;
      m.depthWrite = false;
    }
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
    entry.mesh.scale.setScalar(entry.tileScale / o.s);
    entry.mesh.position.set(
      (entry.tileMx - o.x) / o.s, // east metres from origin
      0,
      (o.y - entry.tileMy) / o.s, // north metres from origin
    );
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
   * Add terrain elevation to each vertex Y: the building's anchor (its centre)
   * everywhere, except that vertices at its base follow the ground down where
   * it drops below the anchor — so the downhill walls of a building on a slope
   * reach the ground instead of floating. (fill-extrusion sinks the base 10 m
   * instead; these meshes aren't hidden by terrain, so that would show.)
   */
  private bake(entry: TileEntry) {
    if (!entry.geometry || !entry.positionBase || !entry.features) return;
    const attr = entry.geometry.getAttribute('position') as THREE.BufferAttribute;
    const pos = attr.array as Float32Array;
    const base = entry.positionBase;
    const elevationAt = this.elevation.batch();
    const groundAt = createMercatorElevationSampler(this.map);
    const scale = entry.tileScale ?? 0;

    for (const b of entry.features) {
      const ele = elevationAt(new LngLat(b.lng, b.lat));
      const end = b.start + b.count;
      for (let v = b.start; v < end; v++) {
        const rel = base[v * 3 + 1];
        pos[v * 3 + 1] =
          rel <= BASE_EPSILON_M && scale
            ? Math.min(ele, groundAt(entry.tileMx + pos[v * 3] * scale, entry.tileMy - pos[v * 3 + 2] * scale)) + rel
            : rel + ele;
      }
    }
    attr.needsUpdate = true;
    entry.geometry.computeBoundingSphere();
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
    if (this.onSourceData) this.map.off('sourcedata', this.onSourceData);
    if (this.rebakeRafId !== undefined) cancelAnimationFrame(this.rebakeRafId);
    this.worker.terminate();
    for (const entry of this.tiles.values()) {
      if (entry.mesh) this.group.remove(entry.mesh);
      entry.geometry?.dispose();
    }
    this.tiles.clear();
    this.scene.remove(this.group);
    this.material.dispose();
  }
}

function toOrigin(lngLat: LngLat): SceneOrigin {
  const om = MercatorCoordinate.fromLngLat(lngLat, 0);
  return { x: om.x, y: om.y, z: om.z, s: om.meterInMercatorCoordinateUnits() };
}

/** Do two geographic bounding boxes intersect? (edge-touching counts.) */
function overlaps(a: GeoBounds, b: GeoBounds): boolean {
  return a.west <= b.east && a.east >= b.west && a.south <= b.north && a.north >= b.south;
}
