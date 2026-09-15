/**
 * Deterministic placement of model instances along lines and inside polygons.
 *
 * Pure geometry — no three.js, no MapLibre. Everything works in Web-Mercator
 * metres, and every generated point is a function of its WORLD POSITION only:
 * never of the feature it came from, its index, or the tile it arrived in. That
 * matters because instances are rebuilt on every `moveend`, and a tiled source
 * hands the same forest to us as differently clipped (and buffer-overlapping)
 * pieces at every viewport. Position-keyed generation makes those pieces agree
 * with each other and with themselves across rebuilds, so the trees hold still.
 */

const R = 6378137;
const D2R = Math.PI / 180;

/** Dedupe/id quantisation: 0.5 m. Two points closer than this are the same tree. */
const QUANT = 2;

/**
 * Key stride for packing a quantised (x, y) into one float. 2^27 exceeds the
 * quantised mercator range (±4.0e7), and the product stays under 2^53, so keys
 * are exact and collision-free worldwide.
 */
const KEY_STRIDE = 134217728;

export interface PlacedPoint {
  lng: number;
  lat: number;
  /** Synthetic feature id: stable, and distinct per generated instance. */
  id: number;
}

/** Clip rectangle in mercator metres. */
export interface MercBounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

/** Receives each placed point; return `false` to stop placing (budget spent). */
export type Emit = (point: PlacedPoint) => boolean;

/** Flat `[x0, y0, x1, y1, …]` in mercator metres. */
export type MercRing = number[];

// ----------------------------------------------------------------- mercator

export const toMercX = (lng: number): number => R * lng * D2R;
export const toMercY = (lat: number): number =>
  R * Math.log(Math.tan(Math.PI / 4 + (lat * D2R) / 2));
export const fromMercX = (x: number): number => x / R / D2R;
export const fromMercY = (y: number): number =>
  (2 * Math.atan(Math.exp(y / R)) - Math.PI / 2) / D2R;

/**
 * Mercator metres per real metre at a latitude, quantised to 0.05° bands.
 *
 * Spacing is specified in real metres but the grid lives in mercator metres, and
 * the two differ by 1/cos(lat) — 1.5x at Vienna. Quantising keeps the factor a
 * pure function of position (so the grid stays global and stable) at the cost of
 * a <0.1% spacing step between bands, which is invisible.
 */
export function mercatorScale(lat: number): number {
  const band = Math.round(lat / 0.05) * 0.05;
  return 1 / Math.cos(band * D2R);
}

/** Convert a GeoJSON ring/line to a flat mercator array. */
export function toMerc(coords: number[][]): MercRing {
  const out: MercRing = new Array(coords.length * 2);
  for (let i = 0; i < coords.length; i++) {
    out[i * 2] = toMercX(coords[i][0]);
    out[i * 2 + 1] = toMercY(coords[i][1]);
  }
  return out;
}

// --------------------------------------------------------------------- hash

/** 32-bit integer hash (xxHash-style finaliser). Same input → same output. */
export function hash32(x: number, y: number): number {
  let h = Math.imul(x | 0, 0x27d4eb2d) ^ Math.imul(y | 0, 0x165667b1);
  h = Math.imul(h ^ (h >>> 15), 0x2545f491);
  h = Math.imul(h ^ (h >>> 13), 0x27d4eb2d);
  return (h ^ (h >>> 16)) >>> 0;
}

/**
 * Stable identity of a position, quantised to 0.5 m. Doubles as the de-dupe key
 * and — hashed — as the synthetic feature id, so both agree by construction.
 */
export function positionKey(mercX: number, mercY: number): number {
  return Math.round(mercX * QUANT) * KEY_STRIDE + Math.round(mercY * QUANT);
}

/** Emit one point unless its 0.5 m cell was already taken. */
function emitAt(px: number, py: number, seen: Set<number>, emit: Emit): boolean {
  const qx = Math.round(px * QUANT);
  const qy = Math.round(py * QUANT);
  const key = qx * KEY_STRIDE + qy;
  if (seen.has(key)) return true; // duplicate from an overlapping tile piece
  seen.add(key);
  return emit({ lng: fromMercX(px), lat: fromMercY(py), id: hash32(qx, qy) });
}

// --------------------------------------------------------------------- line

/**
 * Walk each line by arc length and place a point every `cell` mercator metres.
 *
 * The starting phase is hashed from the line's first vertex rather than being
 * zero, so parallel rows of hedges don't line up with each other. Returns false
 * if `emit` asked to stop.
 */
export function placeAlongLine(
  lines: number[][][],
  cell: number,
  clip: MercBounds,
  seen: Set<number>,
  emit: Emit,
): boolean {
  for (const line of lines) {
    if (line.length < 2) continue;
    const m = toMerc(line);
    const h = hash32(Math.round(m[0] / cell), Math.round(m[1] / cell));
    let next = ((h & 0xffff) / 65536) * cell;
    let travelled = 0;

    for (let i = 2; i < m.length; i += 2) {
      const ax = m[i - 2];
      const ay = m[i - 1];
      const segment = Math.hypot(m[i] - ax, m[i + 1] - ay);
      if (segment === 0) continue;
      while (next <= travelled + segment) {
        const t = (next - travelled) / segment;
        const px = ax + (m[i] - ax) * t;
        const py = ay + (m[i + 1] - ay) * t;
        if (px >= clip.minX && px <= clip.maxX && py >= clip.minY && py <= clip.maxY) {
          if (!emitAt(px, py, seen, emit)) return false;
        }
        next += cell;
      }
      travelled += segment;
    }
  }
  return true;
}

// --------------------------------------------------------------------- fill

/**
 * Scatter points inside a (multi)polygon on a globally aligned jittered grid.
 *
 * Insideness is resolved per grid ROW with a scanline — one even-odd pass over
 * the ring edges yields the x-spans that are inside, so the cost is
 * O(rows x edges) instead of a point-in-polygon test per candidate. Rings are
 * taken together per polygon, which makes holes fall out of the even-odd rule
 * for free.
 *
 * The scanline decides on cell CENTRES, then each point is jittered anywhere
 * within its cell — so a point may sit up to half a cell outside the polygon.
 * For vegetation that reads as a natural, ragged edge, and it saves a per-point
 * containment test.
 */
export function scatterInPolygon(
  polygons: number[][][][],
  cell: number,
  clip: MercBounds,
  seen: Set<number>,
  emit: Emit,
): boolean {
  const crossings: number[] = [];

  for (const rings of polygons) {
    const merc = rings.map(toMerc);
    let minY = Infinity;
    let maxY = -Infinity;
    for (const ring of merc) {
      for (let i = 1; i < ring.length; i += 2) {
        if (ring[i] < minY) minY = ring[i];
        if (ring[i] > maxY) maxY = ring[i];
      }
    }
    minY = Math.max(minY, clip.minY);
    maxY = Math.min(maxY, clip.maxY);
    if (minY > maxY) continue;

    for (let iy = Math.floor(minY / cell); iy <= Math.floor(maxY / cell); iy++) {
      const rowY = (iy + 0.5) * cell;
      crossings.length = 0;
      for (const ring of merc) {
        for (let i = 0, j = ring.length - 2; i < ring.length; j = i, i += 2) {
          const ay = ring[j + 1];
          const by = ring[i + 1];
          if (ay > rowY === by > rowY) continue; // edge doesn't straddle the row
          crossings.push(ring[j] + ((rowY - ay) / (by - ay)) * (ring[i] - ring[j]));
        }
      }
      if (crossings.length < 2) continue;
      crossings.sort((a, b) => a - b);

      for (let k = 0; k + 1 < crossings.length; k += 2) {
        const x0 = Math.max(crossings[k], clip.minX);
        const x1 = Math.min(crossings[k + 1], clip.maxX);
        if (x0 > x1) continue;
        // Cells whose CENTRE falls inside the span.
        const ix1 = Math.floor(x1 / cell - 0.5);
        for (let ix = Math.ceil(x0 / cell - 0.5); ix <= ix1; ix++) {
          const h = hash32(ix, iy);
          const px = (ix + (h & 0xffff) / 65536) * cell;
          const py = (iy + ((h >>> 16) & 0xffff) / 65536) * cell;
          if (!emitAt(px, py, seen, emit)) return false;
        }
      }
    }
  }
  return true;
}

// ------------------------------------------------------------------ budget

/** Total area of a (multi)polygon in mercator m², holes subtracted. */
export function polygonArea(polygons: number[][][][]): number {
  let total = 0;
  for (const rings of polygons) {
    for (let r = 0; r < rings.length; r++) {
      const m = toMerc(rings[r]);
      let twice = 0;
      for (let i = 0, j = m.length - 2; i < m.length; j = i, i += 2) {
        twice += m[j] * m[i + 1] - m[i] * m[j + 1];
      }
      // Ring 0 is the exterior, the rest are holes (GeoJSON / classifyRings).
      total += (r === 0 ? 1 : -1) * Math.abs(twice / 2);
    }
  }
  return Math.max(0, total);
}

/** Total length of a (multi)line in mercator metres. */
export function lineLength(lines: number[][][]): number {
  let total = 0;
  for (const line of lines) {
    const m = toMerc(line);
    for (let i = 2; i < m.length; i += 2) {
      total += Math.hypot(m[i] - m[i - 2], m[i + 1] - m[i - 1]);
    }
  }
  return total;
}
