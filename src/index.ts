// Public entry point for the maplibre-3d-layers plugins.
//
// `Fill3DLayer` — add the MVT3D tiles as a normal MapLibre vector source, render
// LOD1 with fill-extrusion, then add Fill3DLayer referencing the same source. It
// reads the tile template + zoom range from the source and fetches the ORIGINAL
// tiles itself (same URLs as MapLibre → deduped by the browser cache) to decode
// the type-0 3D geometry.
//
//   import { Fill3DLayer } from '@maptoolkit/maplibre-3d-layers';
//   map.addSource('src', { type: 'vector', url: 'https://…/tiles.json?…' });
//   // LOD1 (2D footprints) — a layer must consume the source so MapLibre fetches its tiles:
//   map.addLayer({ id: 'lod1', type: 'fill-extrusion', source: 'src', 'source-layer': 'buildings2d', ... });
//   // Full 3D:
//   map.addLayer(new Fill3DLayer({ id: 'fill-3d', source: 'src', 'source-layer': 'buildings3d' }));
//
// `GltfLayer` — render every point feature of any source as an instanced glTF
// model, picked and transformed with MapLibre expressions.
//
//   import { GltfLayer } from '@maptoolkit/maplibre-3d-layers';
//   map.addSource('vehicles', { type: 'geojson', data: feed });
//   map.addLayer(new GltfLayer({
//     id: 'vehicles-3d', source: 'vehicles',
//     models: { bus: '/models/bus.glb' },
//     paint: { 'model-rotation': [0, 0, ['get', 'bearing']] },
//   }));
export { Fill3DLayer } from './Fill3DLayer';
export type { Fill3DLayerOptions } from './Fill3DLayer';
export { GltfLayer } from './GltfLayer';
export type { GltfLayerOptions } from './GltfLayer';
export type { Placement } from './GltfLayer';
export type { StyleValue, Vec3 } from './expression';
