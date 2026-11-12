import { emitAo, emitTile, parseTile } from './mvt3dGeometry';
import type { AoResult, ParsedTile, TileRequest, TileResult } from './mvt3dGeometry';

const ctx = self as unknown as DedicatedWorkerGlobalScope;

// Tiles whose geometry is out and whose AO data is
// still to come, in the order they were requested, each with its parsed faces
// (kept in memory, so the second pass need not read the tile again). They are
// worked off one by one, each after a short pause, so that tiles waiting for
// their geometry (their requests arrive as messages in between) come first.
const AO_PAUSE_MS = 50;
interface Waiting {
  parsed: ParsedTile;
  seq?: number;
  skirtRadius: number;
  aoRadius: number;
}
const waiting = new Map<string, Waiting>();
let scheduled = false;

function scheduleAo() {
  if (scheduled || !waiting.size) return;
  scheduled = true;
  setTimeout(aoStep, AO_PAUSE_MS);
}

function aoStep() {
  scheduled = false;
  const next = waiting.entries().next();
  if (next.done) return;
  const [key, w] = next.value;
  waiting.delete(key);
  const r = emitAo(w.parsed, w.skirtRadius, w.aoRadius);
  const transfer: ArrayBuffer[] = [r.aoFace.buffer as ArrayBuffer, r.aoUV.buffer as ArrayBuffer, r.columnXZ.buffer as ArrayBuffer];
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
  ctx.postMessage({ key, seq: w.seq, phase: 'ao', ...r } satisfies AoResult, transfer);
  scheduleAo();
}

ctx.onmessage = (e: MessageEvent<TileRequest | { cancel: string; seq?: number }>) => {
  if ('cancel' in e.data) {
    // The tile left the view before its AO was due.
    if (waiting.get(e.data.cancel)?.seq === e.data.seq) waiting.delete(e.data.cancel);
    return;
  }
  const { key, seq, x, y, z, buffer, sourceLayer, filter, extrusion, skirtRadius, ao, aoRadius } = e.data;
  const later = ao && e.data.aoLater === true;
  const parsed = parseTile(x, y, z, buffer, sourceLayer, filter, extrusion);
  const result = later ? emitTile(parsed, 0, false, undefined, aoRadius) : emitTile(parsed, skirtRadius, ao, undefined, aoRadius);
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
  const drawn = result.index.length > 0; // read before the transfer empties the arrays
  ctx.postMessage({ key, seq, ...result } satisfies TileResult, transfer);
  if (later && drawn) {
    waiting.set(key, { parsed, seq, skirtRadius, aoRadius });
    scheduleAo();
  }
};
