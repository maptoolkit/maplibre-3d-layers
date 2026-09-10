import { createPropertyExpression, Color } from '@maplibre/maplibre-gl-style-spec';
import type {
  StylePropertySpecification,
  Feature,
  GlobalProperties,
  ExpressionSpecification,
} from '@maplibre/maplibre-gl-style-spec';

/** A paint value: either a constant or a MapLibre expression. */
export type PaintValue<T> = T | ExpressionSpecification;

/** Per-feature evaluator (zoom + feature properties → value). */
export type ColorEval = (zoom: number, props: Record<string, unknown>) => [number, number, number, number];
export type OpacityEval = (zoom: number, props: Record<string, unknown>) => number;

// Synthetic property specs so createPropertyExpression allows full data-driven +
// zoom expressions and coerces color outputs to Color.
const COLOR_SPEC = {
  type: 'color',
  default: '#ffffff',
  'property-type': 'data-driven',
  expression: { interpolated: true, parameters: ['zoom', 'feature'] },
  transition: false,
} as unknown as StylePropertySpecification;

const OPACITY_SPEC = {
  type: 'number',
  default: 1,
  minimum: 0,
  maximum: 1,
  'property-type': 'data-driven',
  expression: { interpolated: true, parameters: ['zoom', 'feature'] },
  transition: false,
} as unknown as StylePropertySpecification;

function feature(props: Record<string, unknown>): Feature {
  return { type: 3, properties: props } as Feature; // 3 = Polygon
}

/** Compile `fill-3d-color` (constant or expression) into an RGBA evaluator (0..1). */
export function compileColor(
  value: PaintValue<string> | undefined,
  fallback: [number, number, number, number],
): ColorEval {
  if (value === undefined) return () => fallback;

  const compiled = createPropertyExpression(value, 'fill-3d-color', COLOR_SPEC);
  if (compiled.result !== 'success') {
    console.warn('fill-3d-color: invalid expression, using fallback', compiled.value);
    return () => fallback;
  }
  const expr = compiled.value;
  return (zoom, props) => {
    const c = expr.evaluate({ zoom } as GlobalProperties, feature(props)) as Color;
    // Color channels are premultiplied by alpha — undo for a plain RGB triple.
    const inv = c.a > 0 ? 1 / c.a : 0;
    return [c.r * inv, c.g * inv, c.b * inv, c.a];
  };
}

/** Compile `fill-3d-opacity` (constant or expression) into a 0..1 evaluator. */
export function compileOpacity(
  value: PaintValue<number> | undefined,
  fallback: number,
): OpacityEval {
  if (value === undefined) return () => fallback;

  const compiled = createPropertyExpression(value, 'fill-3d-opacity', OPACITY_SPEC);
  if (compiled.result !== 'success') {
    console.warn('fill-3d-opacity: invalid expression, using fallback', compiled.value);
    return () => fallback;
  }
  const expr = compiled.value;
  return (zoom, props) => expr.evaluate({ zoom } as GlobalProperties, feature(props)) as number;
}
