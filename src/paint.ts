import {
  compileColor as compileColorGeneric,
  compileNumber,
} from './expression';
import type { Evaluator, Rgba, StyleValue } from './expression';

/** A paint value: either a constant or a MapLibre expression. */
export type PaintValue<T> = StyleValue<T>;

/** Per-feature evaluator (zoom + feature properties → value). */
export type ColorEval = Evaluator<Rgba>;
export type OpacityEval = Evaluator<number>;

/** Compile `fill-3d-color` (constant or expression) into an RGBA evaluator (0..1). */
export function compileColor(value: PaintValue<string> | undefined, fallback: Rgba): ColorEval {
  return compileColorGeneric(value, 'fill-3d-color', fallback, 3);
}

/** Compile `fill-3d-opacity` (constant or expression) into a 0..1 evaluator. */
export function compileOpacity(value: PaintValue<number> | undefined, fallback: number): OpacityEval {
  return compileNumber(value, 'fill-3d-opacity', fallback, 3);
}
