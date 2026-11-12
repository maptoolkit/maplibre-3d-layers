import { buildTile } from './mvt3dGeometry';
import type { TileRequest, TileResult } from './mvt3dGeometry';

const ctx = self as unknown as DedicatedWorkerGlobalScope;

ctx.onmessage = (e: MessageEvent<TileRequest>) => {
  const { key, x, y, z, buffer, sourceLayer, filter } = e.data;
  const result = buildTile(x, y, z, buffer, sourceLayer, filter);
  const transfer = [
    result.position.buffer,
    result.normal.buffer,
    result.index.buffer,
    result.column.buffer,
    result.columnXZ.buffer,
  ];
  if (result.anchors) {
    const a = result.anchors;
    transfer.push(a.start.buffer, a.column.buffer, a.height.buffer, a.weight.buffer);
  }
  ctx.postMessage({ key, ...result } satisfies TileResult, transfer);
};
