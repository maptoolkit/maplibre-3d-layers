import type { PbfReader as Pbf } from 'pbf';

/**
 * Minimal reader for maptoolkit's z-encoded MVT (`buildings3d` layer).
 *
 * Differs from a standard vector tile in two ways:
 *  - each geometry command carries a THIRD zig-zag varint: the vertex Z (height,
 *    in decimeters),
 *  - command 7 separates polygons of a multipolygon (and appends inner rings).
 *
 * The 3D features have `type === 0` (Unknown); their first polygon is a
 * bounding box (two corner points), not real geometry — callers skip it.
 */

export interface Vertex3D {
  x: number;
  y: number;
  z: number;
}

/** A polygon is a list of rings; a ring is a list of vertices. */
export type Ring = Vertex3D[];
export type Polygon = Ring[];

export class VectorTileFeature {
  properties: Record<string, number | string | boolean> = {};
  extent: number;
  type: 0 | 1 | 2 | 3 = 0;
  id: number | undefined = undefined;

  private _pbf: Pbf;
  private _geometry = -1;
  private _keys: string[];
  private _values: (number | string | boolean)[];

  constructor(
    pbf: Pbf,
    end: number,
    extent: number,
    keys: string[],
    values: (number | string | boolean)[],
  ) {
    this.extent = extent;
    this._pbf = pbf;
    this._keys = keys;
    this._values = values;
    pbf.readFields(readFeature, this, end);
  }

  loadGeometry(): Polygon[] {
    const pbf = this._pbf;
    pbf.pos = this._geometry;

    const end = pbf.readVarint() + pbf.pos;
    let cmd = 1;
    let length = 0;
    let x = 0;
    let y = 0;
    let z = 0;
    let polygon: Polygon = [];
    let ring: Ring | null = null;
    const polygons: Polygon[] = [];

    while (pbf.pos < end) {
      if (length <= 0) {
        const cmdLen = pbf.readVarint();
        cmd = cmdLen & 0x7;
        length = cmdLen >> 3;
      }
      length--;

      if (cmd === 1 || cmd === 2) {
        if (cmd === 1) ring = []; // moveTo
        x += pbf.readSVarint();
        y += pbf.readSVarint();
        z += pbf.readSVarint();
        ring!.push({ x, y, z });
      } else if (cmd === 7) {
        // start a new polygon…
        if (length) {
          polygon.push(ring!);
          polygons.push(polygon);
          polygon = [];
          length--;
          // …or add an inner ring (hole) to the previous polygon
        } else {
          polygons[polygons.length - 1].push(ring!);
        }
      } else {
        throw new Error(`unknown command ${cmd}`);
      }
    }
    return polygons;
  }
}

function readFeature(tag: number, feature: VectorTileFeature, pbf: Pbf) {
  if (tag === 1) feature.id = pbf.readVarint();
  else if (tag === 2) readTag(pbf, feature);
  else if (tag === 3) feature.type = pbf.readVarint() as 0 | 1 | 2 | 3;
  // @ts-expect-error deliberately writing a private field
  else if (tag === 4) feature._geometry = pbf.pos;
}

function readTag(pbf: Pbf, feature: VectorTileFeature) {
  const end = pbf.readVarint() + pbf.pos;
  while (pbf.pos < end) {
    // @ts-expect-error deliberately reading private fields
    const key = feature._keys[pbf.readVarint()] as string;
    // @ts-expect-error deliberately reading private fields
    const value = feature._values[pbf.readVarint()];
    feature.properties[key] = value;
  }
}

export class VectorTileLayer {
  version = 1;
  name = '';
  extent = 4096;
  length = 0;

  private _pbf: Pbf;
  private _keys: string[] = [];
  private _values: (number | string | boolean)[] = [];
  private _features: number[] = [];

  constructor(pbf: Pbf, end?: number) {
    this._pbf = pbf;
    pbf.readFields(readLayer, this, end);
    this.length = this._features.length;
  }

  feature(i: number): VectorTileFeature {
    if (i < 0 || i >= this._features.length) throw new Error('feature index out of bounds');
    this._pbf.pos = this._features[i];
    const end = this._pbf.readVarint() + this._pbf.pos;
    return new VectorTileFeature(this._pbf, end, this.extent, this._keys, this._values);
  }
}

function readLayer(tag: number, layer: VectorTileLayer, pbf: Pbf) {
  if (tag === 15) layer.version = pbf.readVarint();
  else if (tag === 1) layer.name = pbf.readString();
  else if (tag === 5) layer.extent = pbf.readVarint();
  // @ts-expect-error deliberately writing private fields
  else if (tag === 2) layer._features.push(pbf.pos);
  // @ts-expect-error deliberately writing private fields
  else if (tag === 3) layer._keys.push(pbf.readString());
  // @ts-expect-error deliberately writing private fields
  else if (tag === 4) layer._values.push(readValueMessage(pbf));
}

function readValueMessage(pbf: Pbf): number | string | boolean {
  let value: number | string | boolean = 0;
  const end = pbf.readVarint() + pbf.pos;
  while (pbf.pos < end) {
    const tag = pbf.readVarint() >> 3;
    value =
      tag === 1 ? pbf.readString() :
      tag === 2 ? pbf.readFloat() :
      tag === 3 ? pbf.readDouble() :
      tag === 4 ? pbf.readVarint() :
      tag === 5 ? pbf.readVarint() :
      tag === 6 ? pbf.readSVarint() :
      tag === 7 ? pbf.readBoolean() : 0;
  }
  return value;
}

export class VectorTile {
  layers: Record<string, VectorTileLayer>;

  constructor(pbf: Pbf, end?: number) {
    this.layers = pbf.readFields(readTile, {} as Record<string, VectorTileLayer>, end);
  }
}

function readTile(tag: number, layers: Record<string, VectorTileLayer>, pbf: Pbf) {
  if (tag === 3) {
    const layer = new VectorTileLayer(pbf, pbf.readVarint() + pbf.pos);
    if (layer.length) layers[layer.name] = layer;
  }
}
