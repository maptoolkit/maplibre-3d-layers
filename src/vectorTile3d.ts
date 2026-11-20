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
 *
 * `type === 4` is the same geometry as a point table (vts buildings3d,
 * `MTK_BUILDINGS3D_INDEXED_GEOMETRY`): every vertex of the feature once, the
 * rings as lists of vertex numbers — a wall shares its corners with the walls
 * next to it, so a tile is about a fifth smaller. Varints, zig-zag where signed:
 *  - the box: 6 deltas, the first corner from (0, 0, 0), the second from the first,
 *  - V, then V vertices of 3 deltas each, the first from the box's second corner,
 *  - R, then per ring `length * 2 + 1` for an outline (a new polygon) or
 *    `length * 2` for a hole of the polygon before, and `length` deltas of
 *    vertex numbers, each from the number before (the first from 0).
 * `loadGeometry` returns the same {@link Geometry3D} for both.
 */

/**
 * A 3D feature's geometry, flat: a tile has millions of vertices, and an object
 * per vertex costs more than decoding it. Ring r is the vertices `ringEnd[r - 1]`
 * (0 for the first) up to `ringEnd[r]`, polygon p the rings `polygonEnd[p - 1]`
 * up to `polygonEnd[p]` — its outline first, then its holes.
 */
export interface Geometry3D {
  /** x, y, z per vertex in tile units (z: decimetres). */
  xyz: Float64Array;
  ringEnd: number[];
  polygonEnd: number[];
}

export class VectorTileFeature {
  properties: Record<string, number | string | boolean> = {};
  extent: number;
  type: 0 | 1 | 2 | 3 | 4 = 0;
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

  loadGeometry(): Geometry3D {
    if (this.type === 4) return this.loadIndexed();
    const pbf = this._pbf;
    pbf.pos = this._geometry;

    const end = pbf.readVarint() + pbf.pos;
    let cmd = 1;
    let length = 0;
    let x = 0;
    let y = 0;
    let z = 0;
    // A vertex is three varints of at least a byte each: room for all of them.
    const xyz = new Float64Array(end - pbf.pos);
    /** Values written, and where the ring being read starts. */
    let n = 0;
    let ringStart = 0;
    const ringEnd: number[] = [];
    const polygonEnd: number[] = [];

    while (pbf.pos < end) {
      if (length <= 0) {
        const cmdLen = pbf.readVarint();
        cmd = cmdLen & 0x7;
        length = cmdLen >> 3;
      }
      length--;

      if (cmd === 1 || cmd === 2) {
        if (cmd === 1) n = ringStart; // moveTo
        x += pbf.readSVarint();
        y += pbf.readSVarint();
        z += pbf.readSVarint();
        xyz[n++] = x;
        xyz[n++] = y;
        xyz[n++] = z;
      } else if (cmd === 7) {
        ringEnd.push(n / 3);
        ringStart = n;
        // start a new polygon…
        if (length) {
          polygonEnd.push(ringEnd.length);
          length--;
          // …or add an inner ring (hole) to the previous polygon
        } else {
          if (!polygonEnd.length) throw new Error('inner ring without a polygon');
          polygonEnd[polygonEnd.length - 1] = ringEnd.length;
        }
      } else {
        throw new Error(`unknown command ${cmd}`);
      }
    }
    return { xyz: xyz.slice(0, n), ringEnd, polygonEnd };
  }

  /** The point table (`type === 4`, see the top of this file) as {@link Geometry3D}. */
  private loadIndexed(): Geometry3D {
    const pbf = this._pbf;
    pbf.pos = this._geometry;
    const end = pbf.readVarint() + pbf.pos;

    // The box, as the first ring and polygon - like the box in the command encoding.
    const box = new Float64Array(6);
    let x = pbf.readSVarint();
    let y = pbf.readSVarint();
    let z = pbf.readSVarint();
    box[0] = x;
    box[1] = y;
    box[2] = z;
    x += pbf.readSVarint();
    y += pbf.readSVarint();
    z += pbf.readSVarint();
    box[3] = x;
    box[4] = y;
    box[5] = z;
    const vertexCount = pbf.readVarint();
    const vertices = new Float64Array(vertexCount * 3);
    for (let i = 0; i < vertexCount * 3; i += 3) {
      x += pbf.readSVarint();
      y += pbf.readSVarint();
      z += pbf.readSVarint();
      vertices[i] = x;
      vertices[i + 1] = y;
      vertices[i + 2] = z;
    }

    // The vertex numbers first (each a varint of at least a byte: room for all of
    // them), then the vertices they name, into an array of exactly that size.
    const numbers = new Int32Array(end - pbf.pos);
    const heads: number[] = [];
    let count = 0;
    let at = 0;
    for (let r = 0, rings = pbf.readVarint(); r < rings; r++) {
      const head = pbf.readVarint();
      heads.push(head);
      for (let k = head >> 1; k > 0; k--) {
        at += pbf.readSVarint();
        if (at < 0 || at >= vertexCount) throw new Error(`point table: vertex ${at} of ${vertexCount}`);
        numbers[count++] = at;
      }
    }
    const xyz = new Float64Array(6 + count * 3);
    xyz.set(box);
    for (let i = 0, o = 6; i < count; i++, o += 3) {
      const v = numbers[i] * 3;
      xyz[o] = vertices[v];
      xyz[o + 1] = vertices[v + 1];
      xyz[o + 2] = vertices[v + 2];
    }
    const ringEnd: number[] = [2];
    const polygonEnd: number[] = [1];
    let points = 2;
    for (const head of heads) {
      points += head >> 1;
      ringEnd.push(points);
      if (head & 1) polygonEnd.push(ringEnd.length);
      else polygonEnd[polygonEnd.length - 1] = ringEnd.length; // a hole of the polygon before, as in loadGeometry
    }
    if (pbf.pos !== end) throw new Error('point table: length mismatch');
    return { xyz, ringEnd, polygonEnd };
  }

  /**
   * A standard 2D MVT geometry (`buildings2d` polygons, two varints per point,
   * command 7 = ClosePath) as rings, NOT closed by a repeated first point.
   */
  loadRings2D(): { x: number; y: number }[][] {
    const pbf = this._pbf;
    pbf.pos = this._geometry;

    const end = pbf.readVarint() + pbf.pos;
    let cmd = 1;
    let length = 0;
    let x = 0;
    let y = 0;
    const rings: { x: number; y: number }[][] = [];
    let ring: { x: number; y: number }[] | null = null;

    while (pbf.pos < end) {
      if (length <= 0) {
        const cmdLen = pbf.readVarint();
        cmd = cmdLen & 0x7;
        length = cmdLen >> 3;
      }
      length--;

      if (cmd === 1 || cmd === 2) {
        x += pbf.readSVarint();
        y += pbf.readSVarint();
        if (cmd === 1) {
          if (ring) rings.push(ring);
          ring = [];
        }
        ring!.push({ x, y });
      } else if (cmd === 7) {
        if (ring) rings.push(ring);
        ring = null;
      } else {
        throw new Error(`unknown command ${cmd}`);
      }
    }
    if (ring) rings.push(ring);
    return rings;
  }
}

function readFeature(tag: number, feature: VectorTileFeature, pbf: Pbf) {
  if (tag === 1) feature.id = pbf.readVarint();
  else if (tag === 2) readTag(pbf, feature);
  else if (tag === 3) feature.type = pbf.readVarint() as 0 | 1 | 2 | 3 | 4;
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
