import { buildTile } from './mvt3dGeometry';
import type { TileRequest, TileResult } from './mvt3dGeometry';

const ctx = self as unknown as DedicatedWorkerGlobalScope;

ctx.onmessage = (e: MessageEvent<TileRequest>) => {
  const { key, x, y, z, buffer, sourceLayer, filter, extrusion, skirtRadius, ao, aoRadius } = e.data;
  const result = buildTile(x, y, z, buffer, sourceLayer, filter, extrusion, skirtRadius, ao, undefined, aoRadius);
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
  ctx.postMessage({ key, ...result } satisfies TileResult, transfer);
};
