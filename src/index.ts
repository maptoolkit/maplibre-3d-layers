// Public entry point for the fill-3d plugin. Add the MVT3D tiles as a normal
// MapLibre vector source, render LOD1 with fill-extrusion, then add Fill3DLayer
// referencing the same source. It reads the tile template + zoom range from the
// source and fetches the ORIGINAL tiles itself (same URLs as MapLibre → deduped
// by the browser cache) to decode the type-0 3D geometry.
//
//   import { Fill3DLayer } from 'fill3d-three';
//   map.addSource('src', { type: 'vector', url: 'https://…/tiles.json?…' });
//   // LOD1 (2D footprints) — a layer must consume the source so MapLibre fetches its tiles:
//   map.addLayer({ id: 'lod1', type: 'fill-extrusion', source: 'src', 'source-layer': 'buildings2d', ... });
//   // Full 3D:
//   map.addLayer(new Fill3DLayer({ id: 'fill-3d', source: 'src', 'source-layer': 'buildings3d' }));
export { Fill3DLayer } from './Fill3DLayer';
export type { Fill3DLayerOptions } from './Fill3DLayer';
