import { LngLat } from 'maplibre-gl';
import type { Map, CustomLayerInterface, CustomRenderMethodInput } from 'maplibre-gl';
import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { GltfManager } from './GltfManager';
import type { ImpostorLight } from './GltfManager';
import { compileColor, compileNumber, compileString, compileVec3 } from './expression';
import type { StyleValue, Vec3 } from './expression';
import type { ExpressionSpecification, FilterSpecification } from '@maplibre/maplibre-gl-style-spec';

/** How a feature turns into models. `'auto'` picks by geometry type. */
export type Placement = 'auto' | 'point' | 'line' | 'fill';

/** Point features evaluate `['geometry-type']` as Point. */
const POINT = 1 as const;

const DEFAULT_LIGHTING = { ambient: 1.5, directional: 2.5, environment: 0.7 };

/** Metres between generated instances when `model-spacing` is not given. */
const DEFAULT_SPACING = 25;

/**
 * Instances per build before the layer thins itself out. Distant ones draw as
 * billboards by default, so the GPU cost is mostly the ~1–2k full meshes near
 * the camera; what grows with this is the CPU time of a rebuild (~70 ms for
 * 20k trees on a MacBook Air).
 */
const DEFAULT_MAX_INSTANCES = 20000;

/**
 * Default `model-thinning-distance`: full density within ~2.5 km of the camera
 * at z16 (enough for a 60° pitch), halving per zoom level. Base 0.5 between
 * stops 2^8 apart makes it halve exactly with each zoom step.
 */
const DEFAULT_THINNING_DISTANCE: ExpressionSpecification = [
  'interpolate', ['exponential', 0.5], ['zoom'], 12, 40000, 20, 156.25,
];

/**
 * Default `model-impostor-distance`: billboards beyond ~800 m at z16, where a
 * tree-sized model is ~20–30 px tall; halving per zoom level like the above.
 */
const DEFAULT_IMPOSTOR_DISTANCE: ExpressionSpecification = [
  'interpolate', ['exponential', 0.5], ['zoom'], 12, 12800, 20, 50,
];

const PLACEMENTS: readonly Placement[] = ['auto', 'point', 'line', 'fill'];

export interface GltfLayerOptions {
  /** Layer id (required). */
  id: string;
  /** Id of a source already added to the map (required) — `geojson` or `vector`. */
  source: string;
  /** Vector-tile layer to read within the source; required for vector sources. */
  'source-layer'?: string;
  /** Model id → glTF/glb URL (required, at least one). `layout['model-id']` picks. */
  models: Record<string, string>;
  /** MapLibre filter expression; features that fail are not rendered. */
  filter?: FilterSpecification;
  /** Hide the layer below this zoom. */
  minzoom?: number;
  /** Hide the layer at and above this zoom. */
  maxzoom?: number;
  /** raster-dem source id for elevation; defaults to the map's active terrain. */
  terrainSourceId?: string;
  /**
   * Upper bound on instances per rebuild (default `20000`). Denser data is
   * thinned — far away first while `model-thinning-distance` is on.
   */
  maxInstances?: number;
  /**
   * Override the models' `doubleSided` flag: `false` draws front faces only,
   * `true` both. By default a double-sided mesh that is closed (every edge
   * shared by two consistently wound triangles) draws front faces only — its
   * back faces are hidden anyway — and open ones such as leaf cards keep both.
   */
  doubleSided?: boolean;
  /**
   * Light magnitudes. Direction and colour come from the style's `light`
   * property; these only scale it.
   */
  lighting?: {
    /** Ambient fill light (default `1.5`). */
    ambient?: number;
    /** Directional sun light (default `2.5`). */
    directional?: number;
    /** Soft all-round light that glossy and metallic materials need (default `0.7`; `0` = off). */
    environment?: number;
  };
  layout?: {
    /** `'none'` hides the layer. */
    visibility?: 'visible' | 'none';
    /** Which entry of `models` a feature uses. Defaults to the only model. */
    'model-id'?: StyleValue<string>;
    /**
     * How features become instances (default `'auto'`): one model per point,
     * spaced along a line, or scattered inside a polygon. `'auto'` picks by
     * geometry type, so one layer can serve all three. Constant per layer.
     */
    'model-placement'?: Placement;
    /**
     * Metres between generated instances on lines and in polygons (default
     * `25`). Ignored for point placement. Data-driven and zoom-driven.
     */
    'model-spacing'?: StyleValue<number>;
    /**
     * Metres from the camera's ground position up to which instances are placed
     * at full density (default: ~2.5 km at z16, halving per zoom level). Beyond
     * it, density halves every further `model-thinning-distance` metres, and
     * when `maxInstances` is exceeded this distance shrinks instead of thinning
     * the foreground. `Infinity` switches it off. Zoom-driven, not data-driven.
     */
    'model-thinning-distance'?: StyleValue<number>;
    /**
     * Metres from the camera's ground position beyond which models draw as
     * billboards (default: ~800 m at z16, halving per zoom level). Each model is
     * rendered once into a small texture; far instances show it on a
     * camera-facing quad of two triangles instead of the full mesh — a few
     * pixels tall at that distance, but otherwise the bulk of the cost.
     * Models in the area a top-down view of the current zoom shows (around the
     * map centre) always stay full meshes, at any pitch. `Infinity` switches
     * billboards off. Zoom-driven, not data-driven.
     */
    'model-impostor-distance'?: StyleValue<number>;
  };
  paint?: {
    /** Multiplied onto the model's own alpha. Constant, or a zoom/data expression. */
    'model-opacity'?: StyleValue<number>;
    /** Multiplied onto the model's own material colour (default: untinted). */
    'model-color'?: StyleValue<string>;
    /** Degrees around `[east, north, up]` — `[2]` is a compass heading. */
    'model-rotation'?: StyleValue<Vec3> | unknown[];
    /** Per-axis factor on the model's own size. */
    'model-scale'?: StyleValue<Vec3> | unknown[];
    /** Metres along `[east, north, up]`, applied after rotation/scale. */
    'model-translation'?: StyleValue<Vec3> | unknown[];
  };
}

/**
 * A MapLibre custom layer that renders a source's features as glTF models,
 * placed on the map's terrain: one model per point, models spaced along lines,
 * or models scattered across polygons.
 *
 * Features come from a source already added to the map. A `geojson` source works
 * on its own; a `vector` source additionally needs a normal style layer
 * consuming it, so that MapLibre loads its tiles.
 *
 * ```js
 * map.setTerrain({ source: 'terrain' });
 * map.addSource('vehicles', { type: 'geojson', data: feed });
 * map.addLayer(new GltfLayer({
 *   id: 'vehicles-3d',
 *   source: 'vehicles',
 *   models: { bus: '/models/bus.glb', tram: '/models/tram.glb' },
 *   filter: ['==', ['get', 'active'], true],
 *   layout: { 'model-id': ['match', ['get', 'mode'], 'tram', 'tram', 'bus'] },
 *   paint: {
 *     'model-rotation': [0, 0, ['get', 'bearing']],
 *     'model-scale': [1, 1, 1],
 *     'model-opacity': ['interpolate', ['linear'], ['zoom'], 14, 0, 16, 1],
 *   },
 * }));
 * ```
 *
 * Paint and layout values are evaluated per feature (its properties plus the
 * current zoom) whenever the layer refreshes: when the map stops moving, and
 * when the source's data changes. Custom layers have no `setPaintProperty`, so
 * the options themselves are read once when the layer is added.
 */
export class GltfLayer implements CustomLayerInterface {
  readonly id: string;
  readonly type = 'custom' as const;
  readonly renderingMode = '3d' as const;
  readonly minzoom?: number;
  readonly maxzoom?: number;

  private readonly options: GltfLayerOptions;
  private readonly visible: boolean;
  private readonly placement: Placement;
  private readonly lighting: Required<NonNullable<GltfLayerOptions['lighting']>>;

  private map!: Map;
  private camera!: THREE.Camera;
  private scene!: THREE.Scene;
  private renderer!: THREE.WebGLRenderer;
  private ambient!: THREE.HemisphereLight;
  private sun!: THREE.DirectionalLight;
  private manager?: GltfManager;
  private environmentDone = false;

  // Reused every frame to avoid per-frame allocations.
  private readonly mainMat = new THREE.Matrix4();
  private readonly originMat = new THREE.Matrix4();
  private readonly originScale = new THREE.Vector3();
  private readonly sceneToClip = new THREE.Matrix4();
  private readonly clipToScene = new THREE.Matrix4();
  private readonly eye = new THREE.Vector4();
  private readonly eyeShift = new THREE.Matrix4();
  private readonly sunColor = new THREE.Color();

  constructor(options: GltfLayerOptions) {
    if (!options?.id) throw new Error('GltfLayer: `id` is required');
    if (!options.source) throw new Error('GltfLayer: `source` (a source id) is required');
    if (!options.models || Object.keys(options.models).length === 0) {
      throw new Error('GltfLayer: `models` must hold at least one { id: url } entry');
    }

    this.id = options.id;
    this.minzoom = options.minzoom;
    this.maxzoom = options.maxzoom;
    this.options = options;
    this.visible = options.layout?.visibility !== 'none';
    this.lighting = { ...DEFAULT_LIGHTING, ...options.lighting };

    const placement = options.layout?.['model-placement'] ?? 'auto';
    if (!PLACEMENTS.includes(placement)) {
      throw new Error(
        `GltfLayer: model-placement must be one of ${PLACEMENTS.join(', ')} — got '${placement}'`,
      );
    }
    this.placement = placement;
  }

  onAdd(map: Map, gl: WebGL2RenderingContext) {
    this.map = map;
    const terrainSourceId = this.options.terrainSourceId ?? map.getTerrain()?.source ?? '';

    this.camera = new THREE.Camera();
    this.scene = new THREE.Scene();
    // three.js is y-up / z-toward-viewer; rotate + mirror so the scene frame is
    // (x=east, y=up, z=north) to match maplibre. The two flips preserve winding
    // (three flips the front face itself for negative-determinant matrices).
    this.scene.rotateX(Math.PI / 2);
    this.scene.scale.multiply(new THREE.Vector3(1, 1, -1));

    // glTF ships PBR materials, so unlike fill-3d this layer uses real lights.
    this.ambient = new THREE.HemisphereLight(0xffffff, 0x9a9a9a, this.lighting.ambient);
    this.sun = new THREE.DirectionalLight(0xffffff, this.lighting.directional);
    this.scene.add(this.ambient, this.sun, this.sun.target);

    this.renderer = new THREE.WebGLRenderer({
      canvas: map.getCanvas(),
      context: gl,
      antialias: true,
    });
    this.renderer.autoClear = false;

    const paint = this.options.paint ?? {};
    const layout = this.options.layout ?? {};
    const modelIds = Object.keys(this.options.models);

    this.manager = new GltfManager(
      map,
      this.scene,
      LngLat.convert(map.getCenter()), // initial anchor; re-anchors itself on move
      terrainSourceId,
      {
        sourceId: this.options.source,
        sourceLayer: this.options['source-layer'],
        filter: this.options.filter,
        minzoom: this.options.minzoom,
        maxzoom: this.options.maxzoom,
        models: this.options.models,
        // With a single model, `model-id` is redundant — default to it.
        modelId: compileString(layout['model-id'], 'model-id', modelIds[0], POINT),
        placement: this.placement,
        spacing: compileNumber(layout['model-spacing'], 'model-spacing', DEFAULT_SPACING, POINT),
        thinningDistance: compileNumber(
          layout['model-thinning-distance'] ?? DEFAULT_THINNING_DISTANCE, 'model-thinning-distance', Infinity, POINT,
        ),
        impostorDistance: compileNumber(
          layout['model-impostor-distance'] ?? DEFAULT_IMPOSTOR_DISTANCE, 'model-impostor-distance', Infinity, POINT,
        ),
        maxInstances: this.options.maxInstances ?? DEFAULT_MAX_INSTANCES,
        doubleSided: this.options.doubleSided,
        rotation: compileVec3(paint['model-rotation'], 'model-rotation', [0, 0, 0], POINT),
        scale: compileVec3(paint['model-scale'], 'model-scale', [1, 1, 1], POINT),
        translation: compileVec3(paint['model-translation'], 'model-translation', [0, 0, 0], POINT),
        color:
          paint['model-color'] === undefined
            ? undefined
            : compileColor(paint['model-color'], 'model-color', [1, 1, 1, 1], POINT),
        opacity:
          paint['model-opacity'] === undefined
            ? undefined
            : compileNumber(paint['model-opacity'], 'model-opacity', 1, POINT),
      },
    );
  }

  render(_gl: WebGL2RenderingContext, args: CustomRenderMethodInput) {
    if (!this.visible) return; // hidden → don't draw, don't keep repainting

    // Fold the (re-anchored) scene origin into the camera projection so instance
    // matrices stay in small local metres. Origin is at sea level; instances
    // carry their absolute terrain elevation in Y.
    const o = this.manager!.origin;
    this.mainMat.fromArray(args.defaultProjectionData.mainMatrix as unknown as number[]);
    this.originMat
      .makeTranslation(o.x, o.y, o.z)
      .scale(this.originScale.set(o.s, -o.s, o.s));
    this.sceneToClip.multiplyMatrices(this.mainMat, this.originMat);
    this.placeCamera();

    this.syncLight();
    this.renderer.resetState();
    // Built on the live context, so it has to happen inside a render tick —
    // resetState() above already ran, and the render below re-syncs three's state.
    if (!this.environmentDone) this.buildEnvironment();
    // Same for impostors, which also want the environment built above.
    if (this.manager!.needsImpostors()) {
      this.manager!.bakeImpostors(this.renderer, this.impostorLight());
      this.renderer.resetState();
    }
    this.manager!.setViewBearing(this.map.getBearing());
    this.renderer.render(this.scene, this.camera);
    // No triggerRepaint() here: that re-renders the map every frame even when
    // nothing changes. Builds, model loads and DEM updates request their own.
  }

  /**
   * Put three's camera where MapLibre's actually is. Only the projection comes
   * from MapLibre, so the camera would otherwise sit at the scene origin — sea
   * level at the map centre of the last rebuild — and every view-dependent term
   * (specular, environment reflections) would shade as seen from there, and jump
   * whenever a rebuild moves the origin. The eye is where the projection's rays
   * meet: the scene point mapping to clip (0, 0, z, 0). The camera moves there
   * and the projection takes the inverse shift, so positions on screen and
   * frustum culling stay exactly as before.
   */
  private placeCamera() {
    const P = this.sceneToClip;
    this.eye.set(0, 0, 1, 0).applyMatrix4(this.clipToScene.copy(P).invert());
    if (Math.abs(this.eye.w) > 1e-12) {
      this.camera.position.set(this.eye.x / this.eye.w, this.eye.y / this.eye.w, this.eye.z / this.eye.w);
    } else {
      this.camera.position.set(0, 0, 0); // no finite eye (orthographic) — keep the old frame
    }
    const c = this.camera.position;
    this.camera.projectionMatrix.multiplyMatrices(P, this.eyeShift.makeTranslation(c.x, c.y, c.z));
    this.camera.projectionMatrixInverse.copy(this.camera.projectionMatrix).invert();
  }

  /**
   * Point the sun the way MapLibre's own `light` does (call each frame): the
   * style position is spherical in (east, south, up), rotated by the bearing for
   * anchor `viewport`, then mapped into our (east, up, north) frame → (x, z, -y).
   */
  private syncLight() {
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

    // DirectionalLight shines from its position toward its target (the origin).
    this.sun.position.set(east, up, -south).normalize().multiplyScalar(1000);
    this.sunColor.setStyle(light.color ?? '#ffffff', THREE.SRGBColorSpace);
    this.sun.color.copy(this.sunColor);
    this.ambient.color.copy(this.sunColor);
  }

  /**
   * The style light as the viewer sees it, for baking impostors. With anchor
   * `viewport` it stays put relative to the camera, so a bake from the camera's
   * side matches at any bearing; the bake frame is (east, up, toward viewer).
   */
  private impostorLight(): ImpostorLight {
    const light = this.map.getLight() as { position?: [number, number, number] };
    const [r, az, polar] = light.position ?? [1.15, 210, 30];
    const a = ((az + 90) * Math.PI) / 180;
    const p = (polar * Math.PI) / 180;
    return {
      direction: new THREE.Vector3(
        r * Math.cos(a) * Math.sin(p), // east
        r * Math.cos(p), // up
        r * Math.sin(a) * Math.sin(p), // south = toward the viewer
      ),
      color: this.sunColor.clone(),
      ambient: this.lighting.ambient,
      directional: this.lighting.directional,
      environment: this.scene.environment,
      environmentIntensity: this.scene.environmentIntensity,
    };
  }

  /** Synthetic IBL — without one, metallic glTF materials render near-black. */
  private buildEnvironment() {
    this.environmentDone = true;
    if (this.lighting.environment <= 0) return;
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    const room = new RoomEnvironment();
    this.scene.environment = pmrem.fromScene(room, 0.04).texture;
    this.scene.environmentIntensity = this.lighting.environment;
    room.dispose();
    pmrem.dispose();
    this.renderer.resetState(); // PMREM rendered into the shared context
  }

  onRemove() {
    this.manager?.dispose();
    this.scene?.environment?.dispose();
    this.renderer?.dispose();
  }
}
