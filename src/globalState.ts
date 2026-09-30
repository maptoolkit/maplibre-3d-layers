import type { Map } from 'maplibre-gl';
import type { GlobalState } from './expression';

/** The map's global state, or an empty one while the style isn't loaded yet. */
export function globalStateOf(map: Map): GlobalState {
  try {
    return map.getGlobalState();
  } catch {
    return {};
  }
}

/**
 * Call `onChange` whenever one of `refs` changes in the map's global state.
 * `setGlobalStateProperty` always renders a frame, but fires `styledata` only
 * when a native layer uses the key — so every frame compares the values with
 * the ones seen last (a handful of `===`). Returns the unsubscribe.
 */
export function watchGlobalState(map: Map, refs: Iterable<string>, onChange: () => void): () => void {
  const keys = [...new Set(refs)];
  if (keys.length === 0) return () => {};
  const read = () => {
    const state = globalStateOf(map);
    return keys.map((k) => state[k]);
  };
  let last = read();
  const onRender = () => {
    const next = read();
    if (next.every((v, i) => v === last[i])) return;
    last = next;
    onChange();
  };
  map.on('render', onRender);
  return () => map.off('render', onRender);
}
