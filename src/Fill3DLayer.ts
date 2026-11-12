import { LngLat } from 'maplibre-gl';
import type { Map, CustomLayerInterface, CustomRenderMethodInput } from 'maplibre-gl';
import * as THREE from 'three';
import { ElevationCache } from './ElevationCache';
import { Fill3DManager } from './Fill3DManager';
import { compileColor, compileOpacity } from './paint';
import type { PaintValue } from './paint';
import { compileNumber } from './expression';
import type { FilterSpecification } from '@maplibre/maplibre-gl-style-spec';
import type { ExtrusionRequest } from './mvt3dGeometry';

/** Neutral colour used when no `fill-3d-color` is given (white → material shows). */
const NEUTRAL_COLOR: [number, number, number, number] = [1, 1, 1, 1];

export interface Fill3DLayerOptions {
  /** Layer id (required). */
  id: string;
  /** Id of a vector source already added to the map (required) — the tile
   *  template + zoom range are read from it. */
  source: string;
  /** Vector-tile layer to read within the source (required, e.g. `buildings3d`). */
  'source-layer': string;
  /** MapLibre filter expression; features that fail are not rendered. */
  filter?: FilterSpecification;
  /** raster-dem source id for elevation; defaults to the map's active terrain. */
  terrainSourceId?: string;
  /**
   * EXPERIMENTAL: also draw a layer of 2D footprints from the same tiles as
   * extruded prisms (instead of a native `fill-extrusion`), so they get the
   * same fake AO. Roof and wall spans carry `type: 'roof' | 'wall'` for the
   * paint expressions, like the 3D layer's features.
   */
  extrusion?: {
    /** Layer with the footprint polygons, e.g. `buildings2d`. */
    'source-layer': string;
    filter?: FilterSpecification;
    /** Roof height in metres (default `['get', 'height']`). */
    height?: PaintValue<number>;
    /** Bottom in metres (default `['coalesce', ['get', 'min_height'], 0]`). */
    base?: PaintValue<number>;
  };
  layout?: {
    /** `'none'` hides the layer (and skips loading its tiles). */
    visibility?: 'visible' | 'none';
  };
  paint?: {
    /** Constant colour (`'#rgb'`, `'rgba(...)'`) or a MapLibre color expression. */
    'fill-3d-color'?: PaintValue<string>;
    /** Constant 0..1 or a MapLibre number expression. */
    'fill-3d-opacity'?: PaintValue<number>;
    /** EXPERIMENTAL: fake ambient occlusion, 0..1 (default 0 = off). Zoom /
     *  global-state expressions are evaluated each frame. */
    'fill-3d-ambient-occlusion-intensity'?: PaintValue<number>;
    /** EXPERIMENTAL: metres up the wall foot and out from inner edges (default 3). */
    'fill-3d-ambient-occlusion-radius'?: number;
    /** EXPERIMENTAL: width in metres of the darkened ground around the
     *  building feet (default 3; 0 = none). Read when the layer is added. */
    'fill-3d-ambient-occlusion-ground-radius'?: number;
  };
}

/**
 * A MapLibre custom layer that renders maptoolkit's `buildings3d` vector source
 * as full 3D building meshes, draped onto the map's terrain.
 *
 * `id`, `source` (a vector source id) and `source-layer` are required — missing
 * ones throw. Add the vector source first, then this layer:
 *
 * ```js
 * map.setTerrain({ source: 'terrain' });
 * map.addSource('my-src', { type: 'vector', url: 'https://…/tiles.json?…' });
 * // a normal style layer must consume the source so MapLibre loads its tiles,
 * // e.g. a fill-extrusion on the 2D footprints in 'buildings2d'.
 * map.addLayer(new Fill3DLayer({
 *   id: 'fill-3d',
 *   source: 'my-src',
 *   'source-layer': 'buildings3d',
 *   paint: {
 *     'fill-3d-color': ['match', ['get', 'type'], 'roof', '#c33', '#d9d4cc'],
 *     'fill-3d-opacity': ['interpolate', ['linear'], ['zoom'], 14, 0.4, 17, 1],
 *   },
 * }));
 * ```
 *
 * `fill-3d-color` / `fill-3d-opacity` are evaluated per feature (against its
 * feature properties + zoom) into per-vertex colours. `["global-state", …]` reads
 * the map's global state (`map.setGlobalStateProperty`) and updates live; an
 * opacity that is the same for every feature (zoom / global state only) is
 * evaluated each frame, so zoom and pitch fades are smooth. Note: `visibility`
 * and the paint values are read once when the layer is added (no live
 * `setPaintProperty`).
 */
export class Fill3DLayer implements CustomLayerInterface {
  readonly id: string;
  readonly type = 'custom' as const;
  readonly renderingMode = '3d' as const;

  private readonly terrainSourceIdOption?: string;
  private readonly sourceId: string;
  private readonly sourceLayer: string;
  private readonly filter?: FilterSpecification;
  private readonly visible: boolean;
  private readonly colorEval: ReturnType<typeof compileColor>;
  private readonly opacityEval: ReturnType<typeof compileOpacity>;
  private readonly transparent: boolean;
  private readonly extrusion?: ExtrusionRequest;
  private readonly ao: { intensity: ReturnType<typeof compileNumber>; radius: number; groundRadius: number };

  private camera!: THREE.Camera;
  private scene!: THREE.Scene;
  private renderer!: THREE.WebGLRenderer;
  private elevation?: ElevationCache;
  private manager?: Fill3DManager;

  // Reused every frame to avoid per-frame allocations.
  private readonly mainMat = new THREE.Matrix4();
  private readonly originMat = new THREE.Matrix4();
  private readonly originScale = new THREE.Vector3();

  constructor(options: Fill3DLayerOptions) {
    if (!options?.id) throw new Error('Fill3DLayer: `id` is required');
    if (!options.source) throw new Error('Fill3DLayer: `source` (a vector source id) is required');
    if (!options['source-layer']) throw new Error('Fill3DLayer: `source-layer` is required');

    this.id = options.id;
    this.terrainSourceIdOption = options.terrainSourceId;
    this.sourceId = options.source;
    this.sourceLayer = options['source-layer'];
    this.filter = options.filter;
    this.visible = options.layout?.visibility !== 'none';

    const paint = options.paint ?? {};
    this.colorEval = compileColor(paint['fill-3d-color'], NEUTRAL_COLOR);
    this.opacityEval = compileOpacity(paint['fill-3d-opacity'], 1);
    // A transparent material is only needed if opacity may drop below 1.
    this.transparent = 'fill-3d-opacity' in paint;

    const intensity = compileNumber(paint['fill-3d-ambient-occlusion-intensity'], 'fill-3d-ambient-occlusion-intensity', 0);
    if (!intensity.featureConstant) {
      console.warn('Fill3DLayer: fill-3d-ambient-occlusion-intensity must not depend on feature data — ignored');
    }
    this.ao = {
      intensity: intensity.featureConstant ? intensity : compileNumber(0, 'fill-3d-ambient-occlusion-intensity', 0),
      radius: Math.max(0.1, paint['fill-3d-ambient-occlusion-radius'] ?? 3),
      groundRadius: Math.max(0, paint['fill-3d-ambient-occlusion-ground-radius'] ?? 3),
    };

    const ext = options.extrusion;
    if (ext) {
      this.extrusion = { sourceLayer: ext['source-layer'], filter: ext.filter, height: ext.height, base: ext.base };
    }
  }

  onAdd(map: Map, gl: WebGL2RenderingContext) {
    // Use the explicit terrain source, else the map's active terrain, else none.
    const terrainSourceId = this.terrainSourceIdOption ?? map.getTerrain()?.source ?? '';

    this.camera = new THREE.Camera();
    this.scene = new THREE.Scene();
    // three.js is y-up / z-toward-viewer; rotate + mirror so the scene frame is
    // (x=east, y=up, z=north) to match maplibre. The two flips preserve winding.
    this.scene.rotateX(Math.PI / 2);
    this.scene.scale.multiply(new THREE.Vector3(1, 1, -1));
    // No scene lights: the meshes use a custom shader that replicates MapLibre's
    // fill-extrusion directional lighting (see Fill3DManager).

    this.renderer = new THREE.WebGLRenderer({
      canvas: map.getCanvas(),
      context: gl,
      antialias: true,
    });
    this.renderer.autoClear = false;

    this.elevation = new ElevationCache(map, terrainSourceId);
    this.manager = new Fill3DManager(
      map,
      this.scene,
      LngLat.convert(map.getCenter()), // initial anchor; re-anchors itself on pan
      this.elevation,
      terrainSourceId,
      {
        color: this.colorEval,
        opacity: this.opacityEval,
        transparent: this.transparent,
        sourceId: this.sourceId,
        sourceLayer: this.sourceLayer,
        filter: this.filter,
        extrusion: this.extrusion,
        ao: this.ao,
        // The strongest-shade-wins skirt keeps its shade in the framebuffer's alpha.
        skirtMax: gl.getContextAttributes()?.alpha !== false,
      },
    );
    if (this.visible) this.manager.update();
  }

  render(_gl: WebGL2RenderingContext, args: CustomRenderMethodInput) {
    if (!this.visible) return; // hidden → don't draw, don't keep repainting

    // Fold the (possibly re-anchored) scene origin into the camera projection so
    // mesh vertices stay in small local metres. Origin is at sea level; meshes
    // carry their absolute elevation in Y.
    const o = this.manager!.origin;
    this.mainMat.fromArray(args.defaultProjectionData.mainMatrix as unknown as number[]);
    this.originMat
      .makeTranslation(o.x, o.y, o.z)
      .scale(this.originScale.set(o.s, -o.s, o.s));

    // Fully faded out (e.g. a zoom or pitch fade) → nothing to draw.
    if (this.manager!.syncOpacity() <= 0) return;
    this.manager!.syncLight(); // match fill-extrusion shading (light follows bearing)
    this.manager!.syncAo();
    this.camera.projectionMatrix.multiplyMatrices(this.mainMat, this.originMat);
    this.renderer.resetState();
    // A transparent layer: depth pre-pass, then the colour pass — only the
    // frontmost surface blends, so semi-transparent buildings don't show their
    // interior/back walls. An opaque layer: the colour pass alone.
    if (this.manager!.depthPrepass) {
      this.manager!.setPass('depth');
      this.renderer.render(this.scene, this.camera);
    }
    this.manager!.setPass('color');
    this.renderer.render(this.scene, this.camera);
    if (this.manager!.skirtPasses) {
      this.manager!.setPass('skirtCollect');
      this.renderer.render(this.scene, this.camera);
      this.manager!.setPass('skirtApply');
      this.renderer.render(this.scene, this.camera);
    }
    // No triggerRepaint() here: that re-renders the map every frame even when
    // nothing changes. Tile loads and DEM rebakes request their own.
  }

  onRemove() {
    this.manager?.dispose();
    this.elevation?.dispose();
    this.renderer?.dispose();
  }
}
