# @maptoolkit/maplibre-3d-layers

Custom layers for [MapLibre GL JS](https://maplibre.org/) that render real 3D
geometry on top of a normal map, draped onto its terrain. No MapLibre fork — you
add them like any other layer.

| layer | renders |
|-------|---------|
| [`Fill3DLayer`](#fill3dlayer) | maptoolkit's `buildings3d` vector source as full 3D building meshes, shaded to match MapLibre's native `fill-extrusion` |
| [`GltfLayer`](#gltflayer) | a source's features as **glTF models** — one per point, spaced along lines, or scattered across polygons |

Both are configured in MapLibre style-spec shape (`source`, `filter`, `layout`,
`paint`) and evaluate their properties per feature with MapLibre's own expression
engine, so everything you already know about expressions applies.

## Install

`three` and `maplibre-gl` are **peer dependencies** — install them alongside:

```bash
npm install @maptoolkit/maplibre-3d-layers three maplibre-gl
```

```js
import { Fill3DLayer, GltfLayer } from '@maptoolkit/maplibre-3d-layers';
```

### Without a bundler

MapLibre GL JS 6 is ESM-only, so the no-bundler path is an ES-module import map.
A self-contained standalone build ships inside the package, so the import map
needs a **single** entry for `maplibre-gl`:

```html
<link href="https://cdn.jsdelivr.net/npm/maplibre-gl@6.8.0/dist/maplibre-gl.css" rel="stylesheet" />
<script type="importmap">
{ "imports": { "maplibre-gl": "https://cdn.jsdelivr.net/npm/maplibre-gl@6.8.0/dist/maplibre-gl.mjs" } }
</script>
<script type="module">
  import * as maplibregl from 'maplibre-gl';
  import { GltfLayer } from 'https://unpkg.com/@maptoolkit/maplibre-3d-layers/dist/maplibre-3d-layers.standalone.mjs';

  const map = new maplibregl.Map({ /* … */ });
  map.on('load', () => map.addLayer(new GltfLayer({ /* …options… */ })));
</script>
```

Two things to watch: use maplibre-gl's **raw dist path**
(`/dist/maplibre-gl.mjs`), not jsdelivr's `+esm` — the latter re-bundles
maplibre-gl and breaks it. And the standalone build carries its own copy of
three, so don't import three separately on the same page. Pin exact versions in
production.

---

## `Fill3DLayer`

Renders the `buildings3d` vector source as full 3D meshes.

Set it up in three steps:

1. add the vector source,
2. add a normal style layer that **consumes** that source, so MapLibre loads its
   tiles — a `fill-extrusion` on the 2D footprints (`buildings2d`) does the job
   and gives you a simple fallback at low zoom,
3. add `Fill3DLayer` on the `buildings3d` source-layer.

```js
import { Fill3DLayer } from '@maptoolkit/maplibre-3d-layers';

map.on('load', () => {
  // Terrain is optional; the layer picks it up on its own and drapes onto it.
  map.addSource('mtk-terrain', {
    type: 'raster-dem',
    url: 'https://tiles.maptoolkit.org/terrainrgb.json',
    encoding: 'terrarium',
    tileSize: 512,
  });
  map.setTerrain({ source: 'mtk-terrain' });

  map.addSource('buildings3d', {
    type: 'vector',
    url: 'https://dataconnector.maptoolkit.net/maptoolkit/buildings3d2.json?api_key=YOUR_KEY',
  });

  // Footprints — also what makes MapLibre load the source's tiles.
  map.addLayer({
    id: 'buildings-lod1',
    type: 'fill-extrusion',
    source: 'buildings3d',
    'source-layer': 'buildings2d',
    // A multi-part building's outline: its parts are drawn instead.
    filter: ['!=', ['get', 'extrude'], false],
    paint: {
      'fill-extrusion-height': ['get', 'height'],
      'fill-extrusion-color': '#fafafa',
      'fill-extrusion-opacity': 0.9,
    },
  });

  // Full 3D.
  map.addLayer(new Fill3DLayer({
    id: 'fill-3d',
    source: 'buildings3d',
    'source-layer': 'buildings3d',
    paint: {
      'fill-3d-color': ['match', ['get', 'type'], 'roof', '#b0483c', '#d9d4cc'],
      'fill-3d-opacity': 0.9,
    },
  }));
});
```

> Put the `fill-extrusion` on `buildings2d`, **never** on `buildings3d` — a
> native style layer cannot render the 3D source-layer and will produce garbage.

The layer fetches the source's original tiles itself (above `maxzoom` MapLibre
would only hand it re-encoded, overzoomed ones). Those requests go through the
map's `transformRequest` like MapLibre's own, so credentials and headers set
there apply to them too, e.g. a session cookie with `credentials: 'include'`
instead of an `api_key` in the URL.

**Vertical placement.** With terrain, buildings that touch go onto it together:
the source gives their heights in one frame and each building's base (its
lowest point), and the group is fitted to the terrain at all its wall feet
standing on the ground (the median of terrain height − foot height). Neighbours
so stay as they stand to each other in the data, and a building stepping up a
slope isn't lifted by the terrain at its centre. A building without a base
height in the source's frame (bodies generated from OSM) sits on the terrain
height of its centre. Either way the vertices at a building's base follow the
ground down on a slope. The
source also sinks the walls standing on a building's base 10 m below it as a
reserve for slopes (buildings3d `MTK_BUILDINGS3D_GROUND_DEPTH`); the terrain
surface hides that reserve. Without terrain the meshes sit on the flat map,
which hides nothing below it, so the reserve is cut off at the base. Switching
the terrain on or off re-bakes the loaded tiles.

### Options

| option | required | description |
|---|---|---|
| `id` | ✅ | MapLibre layer id |
| `source` | ✅ | id of a vector source already added to the map |
| `source-layer` | ✅ | vector-tile layer to read (e.g. `buildings3d`) |
| `filter` | — | MapLibre filter expression; features that fail are dropped |
| `terrainSourceId` | — | raster-dem source for elevation; defaults to the map's active terrain |
| `layout.visibility` | — | `'visible'` (default) / `'none'` |
| `paint['fill-3d-color']` | — | colour, constant or expression (default white) |
| `paint['fill-3d-opacity']` | — | `0..1`, constant or expression (default `1`) |

Colour and opacity are evaluated per feature against its properties, the
current zoom and the map's [global state](#global-state). An opacity that is the
same for every feature (zoom and global state only) is evaluated every frame.

---

## `GltfLayer`

Renders features as glTF models. Point features get one model each; lines get
models spaced along them; polygons get models scattered across them.

```js
import { GltfLayer } from '@maptoolkit/maplibre-3d-layers';

map.on('load', () => {
  map.addSource('vehicles', { type: 'geojson', data: feed });

  map.addLayer(new GltfLayer({
    id: 'vehicles-3d',
    source: 'vehicles',
    models: {
      bus:  'https://example.com/models/bus.glb',
      tram: 'https://example.com/models/tram.glb',
    },
    filter: ['==', ['get', 'in_service'], true],
    layout: {
      'model-id': ['match', ['get', 'route_type'], 0, 'tram', 'bus'],
    },
    paint: {
      'model-rotation': [0, 0, ['get', 'bearing']],   // heading, cw from north
      'model-scale': [1, 1, 1],
      'model-color': ['case', ['get', 'delayed'], '#ff5a3c', '#ffffff'],
      'model-opacity': ['interpolate', ['linear'], ['zoom'], 12, 0.25, 15, 1],
    },
  }));
});

// Live data: just push new positions, the layer follows.
setInterval(async () => map.getSource('vehicles').setData(await fetchFeed()), 2000);
```

It also works directly off a basemap's **own** vector source — no extra source,
just a filter. This plants a 3D tree on every `natural=tree` point, spaces trees
along every `tree_row`, and scatters them across every `wood` polygon:

```js
const MODELS = { leafy: '/models/tree1.glb', bushy: '/models/tree2.glb' };

// Vary every tree a little, so a row isn't a line of identical clones.
const SIZE = ['+', 0.7, ['/', ['%', ['id'], 9], 12]];   // 0.70 .. 1.37
const LOOK = {
  'model-scale': [SIZE, SIZE, SIZE],
  'model-rotation': [0, 0, ['%', ['id'], 360]],
};

// Single trees — Point geometry, one model each.
map.addLayer(new GltfLayer({
  id: 'trees-single',
  source: 'mtk',                     // the style's own vector source
  'source-layer': 'poi_label',
  filter: ['==', ['get', 'type'], 'tree'],
  minzoom: 15,
  models: MODELS,
  layout: { 'model-id': ['case', ['==', ['%', ['id'], 3], 0], 'bushy', 'leafy'] },
  paint: LOOK,
}));

// Tree rows (lines) AND woods (polygons) in ONE layer: 'model-placement'
// defaults to 'auto', so each feature is handled by its geometry type.
map.addLayer(new GltfLayer({
  id: 'trees-vegetation',
  source: 'mtk',
  'source-layer': 'natural',
  filter: ['in', ['get', 'type'], ['literal', ['tree_row', 'wood']]],
  minzoom: 15,
  models: MODELS,
  layout: {
    'model-id': ['case', ['==', ['%', ['id'], 3], 0], 'bushy', 'leafy'],
    'model-spacing': ['match', ['get', 'type'], 'tree_row', 9, 12],   // metres
  },
  paint: LOOK,
}));
```

That's all it takes, also for a dense forest on a strongly pitched map: by
default the layer places models only where they are on screen, keeps full
density near the camera and thins out towards the horizon, and draws distant
models as billboards — see [Performance and density](#performance-and-density).

### Options

| option | required | description |
|---|---|---|
| `id` | ✅ | MapLibre layer id |
| `source` | ✅ | id of a source already added to the map (`geojson` or `vector`) |
| `models` | ✅ | `{ id: url }` map of glTF/glb files; `layout['model-id']` picks one |
| `source-layer` | — | vector-tile layer to read; **required for vector sources** |
| `filter` | — | MapLibre filter expression; features that fail are dropped |
| `minzoom` / `maxzoom` | — | hidden below `minzoom` and at/above `maxzoom`; outside the range the layer costs nothing |
| `maxInstances` | — | instance budget (default `20000`); denser data is thinned out, far away first, see [Performance and density](#performance-and-density) |
| `terrainSourceId` | — | raster-dem source for elevation; defaults to the map's active terrain |
| `doubleSided` | — | `false` draws front faces only, `true` both; default: automatic — front faces only for closed meshes, both for open ones such as leaf cards |
| `lighting` | — | `{ ambient, directional, environment }`, see [Lighting](#lighting) |
| `layout.visibility` | — | `'visible'` (default) / `'none'` |
| `layout['model-id']` | — | which entry of `models` a feature uses; defaults to the only model |
| `layout['model-placement']` | — | `'auto'` (default) / `'point'` / `'line'` / `'fill'`, see [Placement](#placement) |
| `layout['model-spacing']` | — | metres between models on lines and in polygons (default `25`) |
| `layout['model-thinning-distance']` | — | metres from the camera with full density; beyond, density halves every further such distance (default: ~2.5 km at z16, halving per zoom level; `Infinity` = off), see [Performance and density](#performance-and-density) |
| `layout['model-impostor-distance']` | — | metres from the camera beyond which models draw as billboards (default: ~800 m at z16, halving per zoom level; `Infinity` = off); the area a top-down view shows always stays full models, see [Performance and density](#performance-and-density) |
| `paint['model-scale']` | — | `[x, y, z]` factor on the model's own size (default `[1,1,1]`) |
| `paint['model-rotation']` | — | `[x, y, z]` degrees around `[east, north, up]`; `z` is a compass heading |
| `paint['model-translation']` | — | `[x, y, z]` metres along `[east, north, up]` (default `[0,0,0]`) |
| `paint['model-color']` | — | multiplied onto the model's own material colour (default: untinted) |
| `paint['model-opacity']` | — | multiplied onto the model's own alpha (default `1`) |

### Expressions

Every `layout` and `paint` value accepts a constant or a MapLibre expression over
`['zoom']`, the feature's properties, `['id']` and
[`['global-state', …]`](#global-state). The three-component properties
additionally accept **per-component** expressions, which is the only way to build
a vector out of values — the style spec itself has no syntax for it:

```js
'model-rotation': [0, 0, 90]                    // constant
'model-rotation': ['literal', [0, 0, 90]]       // an expression returning a 3-array
'model-rotation': [0, 0, ['get', 'bearing']]    // per-component
'model-scale': ['match', ['get', 'kind'],
   'tram', ['literal', [1, 1, 1]], ['literal', [2, 2, 2]]]
```

Properties are re-evaluated whenever the layer refreshes (see
[When instances update](#when-instances-update)), so zoom-driven values step at
the end of a zoom rather than continuously. The exception is a `model-opacity`
that is the same for every feature: it is evaluated every frame, so a zoom fade
is smooth.

### Sizing and orientation

`model-scale` is a **factor on the model's own size**, not an absolute size, and
its three components are `[width, height, length]` in the model's own frame.
Well-made glTF assets are modelled at real-world scale, so a 9 m tree at factor
`1` is 9 m tall, and the factor you want is `desired height / model height`:

```js
'model-scale': [1, 1, 1]       // as authored
'model-scale': 2.15            // shorthand: uniform on all three axes
'model-scale': [0.8, 1.6, 0.8] // slimmer and taller
```

A single number is shorthand for all three components, which is what you want
for plain "make it bigger".

Like everything else it can be data- or zoom-driven:

```js
'model-scale': (s => [s, s, s])(
  ['match', ['get', 'type'], 'tree_row', 1.3, 0.9]
)
```

Orientation follows the glTF convention that assets face **+Z** with their right
side at **−X**; the layer maps +Z to north and −X to east. So
`model-rotation: [0, 0, 0]` points a model north, and
`[0, 0, ['get', 'bearing']]` takes a plain compass bearing with no offset.
A georeferenced export with north at −Z and east at +X (Blender's glTF export of
a scene modelled east = +X, north = +Y) needs a 180° turn about the vertical
axis, in the file or as `model-rotation: [0, 0, 180]`. Up to 0.5.3 the layer
mapped +X to east and so drew every model mirrored.
`model-translation` is applied after rotation and scale — use its third component
to lift a model whose origin sits in its middle rather than at its feet.

### Placement

`layout['model-placement']` decides how a feature turns into models. `'auto'`,
the default, picks by geometry type — so **one layer can serve mixed geometry**
from the same source-layer:

| geometry | mode | result |
|---|---|---|
| `Point` / `MultiPoint` | `point` | one model per point; `model-spacing` unused |
| `LineString` / `MultiLineString` | `line` | a model every `model-spacing` metres along the line |
| `Polygon` / `MultiPolygon` | `fill` | models scattered across the area, holes left empty |

Setting it explicitly restricts the layer to that one mode; features whose
geometry doesn't match are skipped with a warning.

Models scattered across a polygon are **stable**: they keep their exact positions
while you pan and zoom, and they are irregular rather than laid out on a visible
grid. Along the edge a model may overhang by up to half the spacing, which keeps
the boundary looking natural rather than cut.

Models along a **line** hold their positions for as long as the source serves the
same geometry. If it hands over a more generalised version of the line at another
zoom, they are placed along the line it actually gets, and can shift.

Each generated model gets its **own stable `id`**, so `['%', ['id'], 9]` and
friends vary per model rather than per source feature — that is what keeps a tree
row from becoming a line of identical clones. Point features keep the id they
have in the source, so `['id']` still addresses the original feature there.

### Performance and density

The defaults are tuned for dense vegetation on a strongly pitched map and
should rarely need changing. What the layer does on its own:

- **Only what's on screen.** Models are placed where the view meets the ground
  (plus a small margin at the sides and bottom), never across the whole
  bounding box of a rotated, pitched view. A large model (over 25 m) stays as
  long as any part of its box is on screen, not only its anchor: a 2 km bridge
  anchored at mid-span still shows at a tower, a tall tower whose foot is
  below the screen still shows its top. Its distance thinning counts from its
  nearest side.
- **Full density near the camera, thinner towards the horizon.** Up to
  `model-thinning-distance` from the camera everything is placed; beyond it the
  density halves every further `model-thinning-distance` — no hard edge.
- **A fixed budget, spent on the foreground.** If the view holds more than
  `maxInstances` models, the thinning distance shrinks, so far models go first.
  Which models remain is decided by a hash of their position: they sit exactly
  where they would at full density, and crossing the budget adds and removes
  models instead of rearranging them.
- **Billboards far away.** Beyond `model-impostor-distance` a model is only a few
  pixels tall, yet its full mesh would be most of the GPU cost. Each model is
  rendered once into a small texture, and distant instances show it on a
  camera-facing quad of two triangles. Colour and opacity carry over (a fade blends
  them like the full models); gloss and reflections don't, which is invisible at that size. Whatever a top-down view of
  the current zoom would show stays a full model, however far you tilt — so at
  0° pitch there are no billboards at all.
- **Cheap to draw.** Closed meshes skip their back faces, and materials only
  blend while `model-opacity` is actually below 1 (e.g. during a zoom fade-in).

With these defaults a forest of 20k trees at 72° pitch renders at 60 fps on a
MacBook Air.

`model-spacing` (metres, data- and zoom-driven) sets the density itself and is
the first thing to adjust for a different look.

#### Advanced tuning

All of these are optional; the defaults are shown.

```js
new GltfLayer({
  // …
  maxInstances: 20000,  // more = denser far away, but each rebuild (on moveend) takes longer
  doubleSided: undefined, // true/false forces both/front faces instead of auto-detecting
  layout: {
    // full density up to this many metres from the camera; ~2.5 km at z16
    'model-thinning-distance': ['interpolate', ['exponential', 0.5], ['zoom'], 12, 40000, 20, 156.25],
    // billboards beyond this; ~800 m at z16, where a tree is ~20–30 px tall
    'model-impostor-distance': ['interpolate', ['exponential', 0.5], ['zoom'], 12, 12800, 20, 50],
  },
});
```

- Set `model-thinning-distance` to `Infinity` to thin uniformly instead (the whole
  view shares the budget, foreground included).
- Set `model-impostor-distance` to `Infinity` to always draw full meshes, e.g. for
  models whose look depends on the viewing angle; lower it for heavy models.
- Rendering cost of the full meshes is triangles × models near the camera, so
  prefer low-poly assets: a thousand copies of a 3.6k-triangle model render at
  60 fps, a thousand copies of a 109k-triangle showcase asset drop to 6 fps.

### Data sources

- **`geojson`** — the whole dataset is used, then narrowed to what's in view. No
  other layer is required, and the layer follows `setData()` / `updateData()`.
- **`vector`** — only features from tiles MapLibre has actually loaded are
  available, so **a normal style layer must consume the source**; otherwise the
  layer warns and stays empty. Pass `source-layer`.

### When instances update

The layer refreshes when the map stops moving and when the source's data changes
(once a burst of tiles has settled, and never mid-move).
Positions jump straight to their new location — there is no interpolation between
updates. `paint`, `layout` and `visibility` are read once when the layer is added;
custom layers have no `setPaintProperty`.

### Lighting

Direction and colour follow the style's `light` property (including
`anchor: 'viewport'`, which rotates with the bearing). The `lighting` option only
scales the magnitudes:

```js
lighting: { ambient: 1.5, directional: 2.5, environment: 0.7 }  // defaults
```

`environment` adds a soft all-round light that metallic and glossy materials need
in order to show anything at all; set it to `0` to switch it off.

---

## Global state

Both layers read MapLibre's global state, like native layers do: set a value with
`map.setGlobalStateProperty()` (or the style's root `state`) and read it with
`['global-state', name]`. MapLibre has no `['pitch']` expression, so camera-driven
styling feeds the camera in from outside:

```js
const syncPitch = () => map.setGlobalStateProperty('pitch', Math.round(map.getPitch()));
map.on('pitch', syncPitch);
syncPitch();

// Faded in by zoom, or by pitch when tilted — the same expression works for
// fill-extrusion-opacity, fill-3d-opacity and model-opacity.
const opacity = ['interpolate', ['linear'], ['zoom'],
  16, ['max', 0, ['/', ['global-state', 'pitch'], 85]],
  17, ['max', 0.3, ['/', ['global-state', 'pitch'], 85]]];
```

How fast a change shows:

- **Opacity that is the same for every feature** (`fill-3d-opacity`,
  `model-opacity` over zoom and global state only): every frame.
- **Other colour and opacity values** (`fill-3d-color`, `model-color`, or an
  opacity that also reads feature data): re-evaluated on the next frame after
  the global state changes.
- **Everything else on `GltfLayer`** (`filter`, `model-id`, scale, rotation,
  placement, …): on the next rebuild — right away while the map is still,
  otherwise when the move ends.

Keep the values coarse (e.g. whole degrees): every `setGlobalStateProperty` is a
style change. With terrain, many of them during an animation cost frames.

## Requirements

MapLibre GL JS **≥6** and three.js **≥0.160**, both peer dependencies. No special
bundler or worker configuration is needed.

## Limitations

**`Fill3DLayer`**

- The `buildings3d` source-layer must not be consumed by a native style layer —
  put `fill-extrusion` on `buildings2d`.
- Meshes are not occluded by terrain in front of them.
- On a strongly pitched map, the nearest foreground may briefly stay flat.

**`GltfLayer`**

- No skinned meshes and no animations — skinned meshes are skipped with a warning.
- Models on a line can shift if the source serves a differently generalised
  version of that line at another zoom. Models in a polygon are unaffected.
- Draco- and KTX2-compressed glTF files are not supported yet.
- Billboards turn about the vertical axis only and are baked once per model from
  one side, so strongly asymmetric models look the same from every direction far
  away. Set `model-impostor-distance` to `Infinity` for those.
- No interpolation between source updates.

**Both**

- `paint`, `layout` and `visibility` are fixed once the layer has been added —
  drive changing values through [global state](#global-state).

## Development

```bash
npm install
npm run dev        # dev server with both demos
npm run typecheck
npm run build
```

`/` renders the `buildings3d` source over Vienna; `/gltf.html` plants 3D trees on
the maptoolkit basemap's own vegetation data. Both use maptoolkit endpoints; the
demo tree models come from [BabylonJS/Assets](https://github.com/BabylonJS/Assets)
(CC-BY-4.0).

## License

MIT
