import { buildTile } from './mvt3dGeometry';
import type { TileRequest, TileResult } from './mvt3dGeometry';

const ctx = self as unknown as DedicatedWorkerGlobalScope;

ctx.onmessage = (e: MessageEvent<TileRequest>) => {
  const { key, x, y, z, buffer, sourceLayer, filter } = e.data;
  const result = buildTile(x, y, z, buffer, sourceLayer, filter);
  ctx.postMessage({ key, ...result } satisfies TileResult, [
    result.position.buffer,
    result.normal.buffer,
    result.index.buffer,
  ]);
};
