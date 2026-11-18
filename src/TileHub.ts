import type { Map } from 'maplibre-gl';
import type { AoResult, TileRequest, TileResult, TileSpec } from './mvt3dGeometry';
import type { LayerZooms } from './dataZooms';
// Inlined as a base64 blob so the built bundle stays single-file — no separate
// worker asset to host. Works for both the ESM package and the <script> build.
import Mvt3dWorker from './mvt3dWorker.ts?worker&inline';

/** The parts of MapLibre's `RequestParameters` a fetch uses. */
type RequestParameters = { url: string; credentials?: RequestCredentials; headers?: HeadersInit };

export interface PoolWorker {
  worker: Worker;
  /** Requests ({@link TileHub.request}'s numbers) it has not answered with the geometry yet. */
  geometry: Set<number>;
  /** Requests whose geometry is out and whose AO data is still to come. */
  ao: Set<number>;
}

/** A layer that gets tiles through a {@link TileHub}. */
export interface TileConsumer {
  onTileParsed(result: TileResult): void;
  onAoParsed(result: AoResult): void;
  /** The tile could not be fetched. */
  onTileFailed(key: string, seq: number): void;
}

interface Want {
  consumer: number;
  seq: number;
  spec: TileSpec;
}

/** A tile on its way to a worker: fetched once for every layer that asked for it in the same task. */
interface Job {
  key: string;
  x: number;
  y: number;
  z: number;
  url: string;
  wants: Want[];
  abort?: AbortController;
}

/** A request the worker has: where it went, and whether AO data follows its geometry. */
interface Sent {
  consumer: number;
  key: string;
  worker: number;
  aoLater: boolean;
}

/** Arriving AO data is handed to its layers within this much main-thread time per frame (ms). */
const ARRIVAL_BUDGET_MS = 4;

const hubs = new WeakMap<Map, globalThis.Map<string, TileHub>>();

/**
 * Loads the tiles of one vector source for the Fill3DLayers on it: a tile is
 * fetched once and passes through a worker once, however many layers read it
 * (each with its own filter — the buildings and the landmarks among them, say),
 * and the layers share the workers. Also knows at which zoom levels the
 * source's layers exist ({@link layerZooms}).
 */
export class TileHub {
  /** The hub of a map's source: one for all layers on it. */
  static for(map: Map, sourceId: string): TileHub {
    let ofMap = hubs.get(map);
    if (!ofMap) hubs.set(map, (ofMap = new globalThis.Map()));
    let hub = ofMap.get(sourceId);
    if (!hub) ofMap.set(sourceId, (hub = new TileHub(map, sourceId)));
    return hub;
  }

  /** Tiles are parsed side by side; a worker keeps a tile until its AO data is out. */
  readonly pool: PoolWorker[] = [];
  private readonly consumers = new globalThis.Map<number, { consumer: TileConsumer; workers: number }>();
  private nextConsumer = 0;
  private seq = 0;
  /** Asked for in this task, by tile. */
  private pending = new globalThis.Map<string, Job>();
  private flushQueued = false;
  /** Being fetched, by request number. */
  private readonly fetching = new globalThis.Map<number, Job>();
  private readonly sent = new globalThis.Map<number, Sent>();
  private readonly arrivals: { run: () => void; geometry: boolean }[] = [];
  private arrivalRaf?: number;
  private zooms?: Promise<LayerZooms | undefined>;
  private disposed = false;

  private constructor(
    private readonly map: Map,
    private readonly sourceId: string,
  ) {}

  /** Returns the layer's number in this hub. `workers`: at most this many tile workers (the largest wish of all layers counts). */
  attach(consumer: TileConsumer, workers: number): number {
    const id = ++this.nextConsumer;
    this.consumers.set(id, { consumer, workers });
    return id;
  }

  /** The layer is gone; the last one takes the workers with it. */
  detach(id: number) {
    this.consumers.delete(id);
    // Whatever is still on its way for this layer is called off (its fetches, its AO passes).
    for (const job of new Set([...this.pending.values(), ...this.fetching.values()])) {
      for (const want of job.wants) if (want.consumer === id) this.cancel(want.seq);
    }
    for (const [seq, s] of [...this.sent]) if (s.consumer === id) this.cancel(seq);
    if (this.consumers.size) return;
    this.disposed = true;
    for (const w of this.pool) w.worker.terminate();
    this.pool.length = 0;
    for (const job of new Set(this.fetching.values())) job.abort?.abort();
    this.fetching.clear();
    this.pending.clear();
    this.arrivals.length = 0;
    if (this.arrivalRaf !== undefined) cancelAnimationFrame(this.arrivalRaf);
    if (hubs.get(this.map)?.get(this.sourceId) === this) hubs.get(this.map)!.delete(this.sourceId);
  }

  /**
   * Ask for a tile for one layer. Layers asking for the same tile in the same
   * task (they follow the same camera) share its fetch and its worker pass.
   * Returns the request's number: it comes back in the results (`seq`) and
   * names the request in {@link cancel}.
   */
  request(consumer: number, key: string, x: number, y: number, z: number, url: string, spec: TileSpec): number {
    const seq = ++this.seq;
    let job = this.pending.get(key);
    if (!job) this.pending.set(key, (job = { key, x, y, z, url, wants: [] }));
    job.wants.push({ consumer, seq, spec });
    if (!this.flushQueued) {
      this.flushQueued = true;
      queueMicrotask(() => this.flush());
    }
    return seq;
  }

  /** The layer no longer wants the tile: not fetched, not parsed for it, or its AO pass dropped — whatever is still ahead. */
  cancel(seq: number) {
    const job = this.fetching.get(seq) ?? [...this.pending.values()].find((j) => j.wants.some((w) => w.seq === seq));
    if (job) {
      job.wants = job.wants.filter((w) => w.seq !== seq);
      this.fetching.delete(seq);
      if (!job.wants.length) {
        job.abort?.abort();
        if (this.pending.get(job.key) === job) this.pending.delete(job.key);
      }
      return;
    }
    const s = this.sent.get(seq);
    if (!s) return;
    this.sent.delete(seq);
    const w = this.pool[s.worker];
    if (!w) return;
    w.geometry.delete(seq);
    w.ao.delete(seq);
    w.worker.postMessage({ cancel: s.key, consumer: s.consumer, seq });
  }

  /**
   * At which zoom levels the source's tiles carry each of its layers: the
   * `vector_layers` of its TileJSON (MapLibre keeps only their names, so it is
   * fetched again — from the browser's cache, as a rule). Undefined when the
   * source has no TileJSON URL or does not list its layers.
   */
  layerZooms(): Promise<LayerZooms | undefined> {
    return (this.zooms ??= (async () => {
      try {
        const url = (this.map.getSource(this.sourceId) as { url?: unknown } | undefined)?.url;
        if (typeof url !== 'string') return undefined;
        const req = await this.transformed(url, 'Source');
        const r = await fetch(req.url, { credentials: req.credentials, headers: req.headers });
        if (!r.ok) return undefined;
        const layers = ((await r.json()) as { vector_layers?: { id?: unknown; minzoom?: unknown; maxzoom?: unknown }[] }).vector_layers;
        if (!Array.isArray(layers)) return undefined;
        const zooms = new globalThis.Map<string, [number, number]>();
        for (const l of layers) {
          if (typeof l?.id !== 'string') continue;
          zooms.set(l.id, [typeof l.minzoom === 'number' ? l.minzoom : -Infinity, typeof l.maxzoom === 'number' ? l.maxzoom : Infinity]);
        }
        return zooms;
      } catch {
        return undefined;
      }
    })());
  }

  private flush() {
    this.flushQueued = false;
    const jobs = this.pending;
    this.pending = new globalThis.Map();
    for (const job of jobs.values()) void this.fetch(job);
  }

  /** Fetch a canonical tile's original pbf and hand it to a worker. */
  private async fetch(job: Job) {
    job.abort = new AbortController();
    for (const w of job.wants) this.fetching.set(w.seq, job);
    let buffer: ArrayBuffer | undefined;
    try {
      const req = await this.transformed(job.url, 'Tile');
      const r = await fetch(req.url, { credentials: req.credentials, headers: req.headers, signal: job.abort.signal });
      if (r.ok) buffer = await r.arrayBuffer();
    } catch {
      // aborted or failed: no buffer
    }
    for (const w of job.wants) this.fetching.delete(w.seq);
    if (this.disposed || !job.wants.length) return;
    if (!buffer) {
      for (const w of job.wants) this.consumers.get(w.consumer)?.consumer.onTileFailed(job.key, w.seq);
      return;
    }
    const index = this.pickWorker();
    const w = this.pool[index];
    for (const want of job.wants) {
      w.geometry.add(want.seq);
      this.sent.set(want.seq, { consumer: want.consumer, key: job.key, worker: index, aoLater: want.spec.ao && want.spec.aoLater === true });
    }
    const request: TileRequest = {
      key: job.key,
      x: job.x,
      y: job.y,
      z: job.z,
      buffer,
      consumers: job.wants.map((want) => ({ ...want.spec, consumer: want.consumer, seq: want.seq })),
    };
    w.worker.postMessage(request, [buffer]);
  }

  /**
   * A tile request as MapLibre sends its own: through the map's `transformRequest`
   * (rewritten URL, credentials such as a session cookie, headers). `_requestManager`
   * is MapLibre-internal; without it, or if the transform fails, the URL is fetched as is.
   */
  private async transformed(url: string, type: 'Tile' | 'Source'): Promise<RequestParameters> {
    const manager = (this.map as unknown as {
      _requestManager?: { transformRequest(url: string, type: string): RequestParameters | Promise<RequestParameters> };
    })._requestManager;
    if (!manager) return { url };
    try {
      return await manager.transformRequest(url, type);
    } catch {
      return { url };
    }
  }

  /**
   * The worker for the next tile — an idle one, else a
   * new one while the pool may grow, else the one with the least to do (tiles
   * waiting for their geometry weigh more than AO passes, which give way to them).
   */
  private pickWorker(): number {
    let best = -1;
    let bestLoad = Infinity;
    this.pool.forEach((w, i) => {
      const load = w.geometry.size * 1000 + w.ao.size;
      if (load < bestLoad) {
        best = i;
        bestLoad = load;
      }
    });
    let limit = 1;
    for (const c of this.consumers.values()) limit = Math.max(limit, c.workers);
    if (best >= 0 && (bestLoad === 0 || this.pool.length >= limit)) return best;

    const index = this.pool.length;
    const w: PoolWorker = { worker: new Mvt3dWorker(), geometry: new Set(), ao: new Set() };
    w.worker.onmessage = (e: MessageEvent<TileResult | AoResult>) => this.onResult(w, e.data);
    this.pool.push(w);
    return index;
  }

  private onResult(w: PoolWorker, result: TileResult | AoResult) {
    const seq = result.seq!;
    const s = this.sent.get(seq);
    const isAo = 'phase' in result;
    if (isAo) w.ao.delete(seq);
    else w.geometry.delete(seq);
    if (!s) return; // cancelled meanwhile
    const consumer = this.consumers.get(s.consumer)?.consumer;
    if (!isAo && s.aoLater && result.index.length > 0) w.ao.add(seq); // its AO pass is due in that worker
    else this.sent.delete(seq);
    if (!consumer) return;
    this.arrive(() => (isAo ? consumer.onAoParsed(result) : consumer.onTileParsed(result)), !isAo);
  }

  /**
   * Hand an arrived tile to its layer, in turn, one tile's geometry per frame:
   * the tiles of a view arrive in a burst, and each is drawn for the first time
   * (its buffers uploaded) in the frame after.
   */
  private arrive(run: () => void, geometry: boolean) {
    this.arrivals.push({ run, geometry });
    if (this.arrivalRaf !== undefined) return;
    const step = () => {
      this.arrivalRaf = undefined;
      const t0 = performance.now();
      while (this.arrivals.length) {
        const next = this.arrivals.shift()!;
        next.run();
        // One tile's geometry per frame; AO data (onto geometry that is drawn already) while there is time.
        if (next.geometry || performance.now() - t0 >= ARRIVAL_BUDGET_MS) break;
      }
      if (this.arrivals.length) this.arrivalRaf = requestAnimationFrame(step);
    };
    this.arrivalRaf = requestAnimationFrame(step);
  }
}
