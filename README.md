# fill3d-three

A [MapLibre GL JS](https://maplibre.org/) **custom layer** that renders a z-encoded
vector tile source (maptoolkit's `buildings3d`) as **real 3D meshes** via
[three.js](https://threejs.org/), draped onto the map's terrain and shaded to
match MapLibre's native `fill-extrusion`.

It is an external re-implementation of a forked MapLibre `fill-3d` layer type — no
MapLibre fork required, just a custom layer you add on top of a normal map.

## Why a custom decoder

The `buildings3d` source is a **non-standard MVT**:

- geometry features are **type 0** (`Unknown`) — MapLibre's renderer ignores them;
- each vertex carries a **third zig-zag varint** (Z / height, in decimetres) that
  the standard 2-coordinate MVT reader can't parse;
- the first polygon of each feature is a **bounding box** (base Z + centroid), not
  real geometry.

So the tiles are decoded and triangulated in a Web Worker (`src/vectorTile3d.ts`
+ `src/mvt3dGeometry.ts`) and rendered with three.js instead of by MapLibre.

## Install & run the demo

```bash
npm install
npm run dev        # vite dev server → http://localhost:5173
npm run typecheck  # tsc --noEmit
```

The demo (`index.html`) loads terrain + the `buildings3d` source over Vienna. It
uses maptoolkit endpoints (`?api_key=iib`).

## Usage

`Fill3DLayer` reads its tile template + zoom range from an **existing MapLibre
vector source** and fetches the original tiles itself. So:

1. add the vector source (via its TileJSON `url`),
2. add a layer that **consumes** the source so MapLibre actually fetches its tiles
   — e.g. a `fill-extrusion` on the 2D footprints (`buildings2d`), which also
   gives you a LOD1 fallback at low zoom,
3. add the `Fill3DLayer` on the `buildings3d` source-layer.

```js
import { Map } from 'maplibre-gl';
import { Fill3DLayer } from 'fill3d-three';

const TILEJSON = 'https://vtc.maptoolkit.net/buildings3d.json?api_key=iib';

map.on('load', () => {
  // Terrain (Fill3DLayer drapes the meshes onto it and picks it up automatically).
  map.addSource('mtk-terrain', {
    type: 'raster-dem',
    url: 'https://tiles.maptoolkit.org/terrainrgb.json',
    encoding: 'terrarium',
    tileSize: 512,
  });
  map.setTerrain({ source: 'mtk-terrain' });

  // One vector source for both LOD1 and 3D.
  map.addSource('buildings3d', { type: 'vector', url: TILEJSON });

  // LOD1: native fill-extrusion on the 2D footprints (and it drives the fetch).
  map.addLayer({
    id: 'buildings-lod1',
    type: 'fill-extrusion',
    source: 'buildings3d',
    'source-layer': 'buildings2d',
    paint: {
      'fill-extrusion-height': ['get', 'height'],
      'fill-extrusion-color': '#fafafa',
      'fill-extrusion-opacity': 0.9,
    },
  });

  // Full 3D.
  map.addLayer(
    new Fill3DLayer({
      id: 'fill-3d',
      source: 'buildings3d',       // the vector source id above
      'source-layer': 'buildings3d',
      paint: {
        'fill-3d-color': ['match', ['get', 'type'], 'roof', '#b0483c', '#d9d4cc'],
        'fill-3d-opacity': 0.9,
      },
    }),
  );
});
```

> The same tile URLs are requested by both MapLibre (for LOD1) and the layer, so
> the browser HTTP cache de-dupes them — effectively a single network fetch.

### Options

| option           | required | description |
|------------------|----------|-------------|
| `id`             | ✅       | MapLibre layer id |
| `source`         | ✅       | id of a vector source already added to the map |
| `source-layer`   | ✅       | vector-tile layer to read (e.g. `buildings3d`) |
| `terrainSourceId`| —        | raster-dem source for elevation; defaults to the map's active terrain |
| `filter`         | —        | MapLibre filter expression; features that fail are dropped |
| `layout.visibility` | —     | `'visible'` (default) / `'none'` (hides the layer & skips tile loading) |
| `paint['fill-3d-color']`   | — | constant colour or a MapLibre **color expression** (default white) |
| `paint['fill-3d-opacity']` | — | constant `0..1` or a MapLibre **number expression** (default `1`) |

`fill-3d-color` / `fill-3d-opacity` are evaluated **per feature** (against its
properties + zoom) into per-vertex colours, using MapLibre's own expression
engine. Note: `paint` and `visibility` are read **once** when the layer is added
(custom layers have no `setPaintProperty`).

## How it works

- **Decode off-thread** — `map.coveringTiles()` picks the tiles; each original
  `.pbf` is fetched and decoded + triangulated (earcut, per-ring Newell normals)
  in a Web Worker into one merged `BufferGeometry` per tile.
- **Worldwide float precision** — geometry is stored in metres relative to each
  tile's own corner; a single scene origin is folded into the camera projection
  matrix and **re-anchored** as you pan, so vertices stay small (no z-fighting,
  works flying e.g. Vienna → Berlin).
- **Terrain** — each feature's base height is baked from
  `map.queryTerrainElevation` at its centroid and re-baked as DEM tiles stream in.
- **Looks like fill-extrusion** — a small custom shader replicates MapLibre's
  fill-extrusion directional lighting (in sRGB, using the style `light` + bearing);
  backface culling + a depth pre-pass keep semi-transparent buildings from showing
  their interior walls.

### Module overview

| file | role |
|------|------|
| `src/Fill3DLayer.ts`   | the `CustomLayerInterface` — scene/camera/renderer, options, per-frame render |
| `src/Fill3DManager.ts` | tile lifecycle, meshes, material/shader, terrain baking, origin rebasing |
| `src/mvt3dWorker.ts`   | thin Web Worker wrapper |
| `src/mvt3dGeometry.ts` | pure decode + triangulate → tile-local metre geometry |
| `src/vectorTile3d.ts`  | z-encoded MVT reader |
| `src/paint.ts`         | compile `fill-3d-*` constants/expressions via the style-spec |
| `src/ElevationCache.ts`| cached `queryTerrainElevation`, invalidated per DEM tile |
| `src/index.ts`         | public entry (`Fill3DLayer`, `Fill3DLayerOptions`) |

## Limitations

- The `buildings3d` source-layer must **not** be consumed by a native style layer
  — MapLibre would mis-parse the type-0/z geometry (and above `maxzoom` its
  client-side overzoom re-encodes tiles and drops the layer). Put `fill-extrusion`
  on `buildings2d`, not `buildings3d`.
- No real terrain-depth **occlusion** of meshes behind hills (the fork sampled an
  internal depth texture; not exposed via the public API).
- On a strongly pitched map, foreground tiles one zoom deeper than the map zoom
  may not yet render 3D — a known LOD gap.
- `paint` / `visibility` are static after `addLayer`.

## Requirements

MapLibre GL JS **6.x**, three.js **0.185**, a bundler that handles ES-module Web
Workers (Vite config included; `optimizeDeps.exclude: ['maplibre-gl']` is needed).
