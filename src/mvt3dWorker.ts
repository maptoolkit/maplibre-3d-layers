import { bakeFlat, emitAo, emitTile, flatSkirt, parseTile, prepare, readTile } from './mvt3dGeometry';
import type { AoResult, FlatSpans, ParsedTile, TileRequest, TileResult } from './mvt3dGeometry';

const ctx = self as unknown as DedicatedWorkerGlobalScope;

// A tile is read once for all layers of its request: each gets its own
// geometry (its filter, its extrusion) from the same opened tile.
//
// Tiles whose geometry is out and whose AO data is
// still to come, in the order they were requested, each with its parsed faces
// (kept in memory, so the second pass need not read the tile again). They are
// worked off one by one, each after a short pause, so that tiles waiting for
// their geometry (their requests arrive as messages in between) come first.
const AO_PAUSE_MS = 50;
interface Waiting {
  key: string;
  consumer: number;
  parsed: ParsedTile;
  seq: number;
  skirtRadius: number;
  aoRadius: number;
  /** A map without terrain: what the heights above ground are computed from (see bakeFlat). */
  flat?: { baseY: Float32Array; spans: FlatSpans };
}
const waiting = new Map<string, Waiting>();
let scheduled = false;

const waitingKey = (consumer: number, key: string) => `${consumer}:${key}`;

function scheduleAo() {
  if (scheduled || !waiting.size) return;
  scheduled = true;
  setTimeout(aoStep, AO_PAUSE_MS);
}

function aoStep() {
  scheduled = false;
  const next = waiting.entries().next();
  if (next.done) return;
  const [id, w] = next.value;
  waiting.delete(id);
  const { column, ...r } = emitAo(w.parsed, w.skirtRadius, w.aoRadius);
  const result: AoResult = { key: w.key, consumer: w.consumer, seq: w.seq, phase: 'ao', ...r };
  const transfer: ArrayBuffer[] = [r.aoFace.buffer as ArrayBuffer, r.aoUV.buffer as ArrayBuffer, r.columnXZ.buffer as ArrayBuffer];
  if (w.flat) {
    result.aoh = new Int16Array(w.flat.baseY.length);
    bakeFlat(undefined, result.aoh, w.flat.baseY, w.flat.spans, column, r.columnXZ.length / 2);
    transfer.push(result.aoh.buffer as ArrayBuffer);
    if (r.skirt) {
      result.skirtSphere = flatSkirt(r.skirt);
      transfer.push(result.skirtSphere.buffer as ArrayBuffer);
    }
  }
  if (r.aoTables) {
    const t = r.aoTables;
    transfer.push(t.faces.buffer as ArrayBuffer, t.cells.buffer as ArrayBuffer, t.lists.buffer as ArrayBuffer, t.edges.buffer as ArrayBuffer);
  }
  if (r.skirt) {
    transfer.push(
      r.skirt.position.buffer as ArrayBuffer,
      r.skirt.coord.buffer as ArrayBuffer,
      r.skirt.index.buffer as ArrayBuffer,
      r.skirt.column.buffer as ArrayBuffer,
    );
  }
  ctx.postMessage(result, transfer);
  scheduleAo();
}

ctx.onmessage = (e: MessageEvent<TileRequest | { cancel: string; consumer: number; seq: number }>) => {
  if ('cancel' in e.data) {
    // The tile left the view before its AO was due.
    const id = waitingKey(e.data.consumer, e.data.cancel);
    if (waiting.get(id)?.seq === e.data.seq) waiting.delete(id);
    return;
  }
  const { key, x, y, z, buffer, consumers } = e.data;
  const reader = readTile(buffer);
  for (const c of consumers) {
    const later = c.ao && c.aoLater === true;
    const parsed = parseTile(x, y, z, reader, c.sourceLayer, c.filter, c.extrusion);
    const built = later ? emitTile(parsed, 0, false, undefined, c.aoRadius) : emitTile(parsed, c.skirtRadius, c.ao, undefined, c.aoRadius);
    const result: TileResult = { key, consumer: c.consumer, seq: c.seq, ...built };
    const drawn = result.index.length > 0;
    let flat: Waiting['flat'];
    if (drawn) {
      const prepared = (result.prepared = prepare(result, c.prep));
      // The AO pass computes the heights above ground from these (they are about to be transferred).
      if (later && c.prep.flat) {
        flat = { baseY: prepared.baseY.slice(), spans: { start: prepared.table.start.slice(), count: prepared.table.count.slice(), rigid: prepared.table.rigid.slice() } };
      }
    }
    const transfer: ArrayBuffer[] = [
      result.position.buffer as ArrayBuffer,
      result.normal.buffer as ArrayBuffer,
      result.aoh.buffer as ArrayBuffer,
      result.aoFace.buffer as ArrayBuffer,
      result.aoUV.buffer as ArrayBuffer,
      result.index.buffer as ArrayBuffer,
      result.column.buffer as ArrayBuffer,
      result.columnXZ.buffer as ArrayBuffer,
    ];
    if (result.prepared) {
      const p = result.prepared;
      const t = p.table;
      transfer.push(
        p.baseY.buffer as ArrayBuffer,
        p.color.buffer as ArrayBuffer,
        t.start.buffer as ArrayBuffer,
        t.count.buffer as ArrayBuffer,
        t.lng.buffer as ArrayBuffer,
        t.lat.buffer as ArrayBuffer,
        t.anchor.buffer as ArrayBuffer,
        t.base.buffer as ArrayBuffer,
        t.rigid.buffer as ArrayBuffer,
      );
      if (p.flat) {
        transfer.push(p.flat.sphere.buffer as ArrayBuffer);
        if (p.flat.skirtSphere) transfer.push(p.flat.skirtSphere.buffer as ArrayBuffer);
      }
    }
    if (result.anchors) {
      const a = result.anchors;
      transfer.push(a.start.buffer as ArrayBuffer, a.column.buffer as ArrayBuffer, a.height.buffer as ArrayBuffer, a.weight.buffer as ArrayBuffer);
    }
    if (result.aoTables) {
      const t = result.aoTables;
      transfer.push(t.faces.buffer as ArrayBuffer, t.cells.buffer as ArrayBuffer, t.lists.buffer as ArrayBuffer, t.edges.buffer as ArrayBuffer);
    }
    if (result.skirt) {
      transfer.push(
        result.skirt.position.buffer as ArrayBuffer,
        result.skirt.coord.buffer as ArrayBuffer,
        result.skirt.index.buffer as ArrayBuffer,
        result.skirt.column.buffer as ArrayBuffer,
      );
    }
    ctx.postMessage(result, transfer);
    if (later && drawn) {
      waiting.set(waitingKey(c.consumer, key), { key, consumer: c.consumer, parsed, seq: c.seq, skirtRadius: c.skirtRadius, aoRadius: c.aoRadius, flat });
      scheduleAo();
    }
  }
};
