import { normalizePropertyExpression, isExpression } from '@maplibre/maplibre-gl-style-spec';
import type {
  Color,
  StylePropertySpecification,
  Feature,
  GlobalProperties,
  ExpressionSpecification,
} from '@maplibre/maplibre-gl-style-spec';

/** A style value: either a constant or a MapLibre expression. */
export type StyleValue<T> = T | ExpressionSpecification;

/** The map's global state (`map.getGlobalState()`), read by `["global-state", …]`. */
export type GlobalState = Record<string, unknown>;

/**
 * A compiled property: evaluate against zoom + a feature's properties/id, and
 * the map's global state for `["global-state", …]`.
 */
export interface Evaluator<T> {
  (zoom: number, props: Record<string, unknown>, id?: string | number, globalState?: GlobalState): T;
  /** Same value for every feature — depends on zoom and global state only. */
  readonly featureConstant: boolean;
  /** Global-state keys the value reads; a change to one of them needs a re-evaluation. */
  readonly globalStateRefs: ReadonlySet<string>;
}

/** MVT geometry type used when evaluating (`['geometry-type']` depends on it). */
export type GeometryType = 1 | 2 | 3; // Point | LineString | Polygon

export type Rgba = [number, number, number, number];
export type Vec3 = [number, number, number];

// Synthetic property specs. MapLibre compiles style properties against the spec
// entry of the property being set; our properties are not in the spec, so we
// supply equivalent ones — `data-driven` + `['zoom','feature']` parameters is
// what enables full zoom *and* per-feature expressions.
function spec(extra: Record<string, unknown>): StylePropertySpecification {
  return {
    'property-type': 'data-driven',
    expression: { interpolated: true, parameters: ['zoom', 'feature'] },
    transition: false,
    ...extra,
  } as unknown as StylePropertySpecification;
}

const COLOR_SPEC = spec({ type: 'color', default: '#ffffff' });
const NUMBER_SPEC = spec({ type: 'number', default: 0 });
const VEC3_SPEC = spec({ type: 'array', value: 'number', length: 3, default: [0, 0, 0] });
// Strings can't be interpolated — `['match'|'case'|'step', …]` still work.
const STRING_SPEC = spec({
  type: 'string',
  default: '',
  expression: { interpolated: false, parameters: ['zoom', 'feature'] },
});

interface Compiled {
  evaluate(globals: GlobalProperties, feature: Feature): unknown;
  /** 'constant' | 'camera' don't read feature data ('source' | 'composite' do). */
  kind: string;
  globalStateRefs?: Set<string>;
}

const NO_REFS: ReadonlySet<string> = new Set();

/** Attach what an evaluator depends on (see {@link Evaluator}). */
function evaluator<T>(
  fn: (zoom: number, props: Record<string, unknown>, id?: string | number, globalState?: GlobalState) => T,
  featureConstant: boolean,
  globalStateRefs: ReadonlySet<string> = NO_REFS,
): Evaluator<T> {
  return Object.assign(fn, { featureConstant, globalStateRefs });
}

/** An evaluator that always returns `value`. */
function constant<T>(value: T): Evaluator<T> {
  return evaluator(() => value, true);
}

/** Wrap a compiled expression; `read` post-processes its raw result. */
function fromCompiled<T>(
  expr: Compiled,
  geometryType: GeometryType,
  read: (v: unknown) => T,
): Evaluator<T> {
  return evaluator(
    (zoom, props, id, globalState) =>
      read(expr.evaluate({ zoom, globalState } as GlobalProperties, makeFeature(geometryType, props, id))),
    expr.kind === 'constant' || expr.kind === 'camera',
    expr.globalStateRefs ?? NO_REFS,
  );
}

/** Compile one value; returns undefined (after warning) if it doesn't parse. */
function compile(
  value: unknown,
  name: string,
  propertySpec: StylePropertySpecification,
): Compiled | undefined {
  try {
    return normalizePropertyExpression(value as never, name, propertySpec) as Compiled;
  } catch (err) {
    console.warn(`${name}: invalid expression, using default`, err);
    return undefined;
  }
}

function makeFeature(
  geometryType: GeometryType,
  props: Record<string, unknown>,
  id?: string | number,
): Feature {
  return { type: geometryType, properties: props, id } as Feature;
}

/** Compile a colour property into an RGBA evaluator with channels in 0..1. */
export function compileColor(
  value: StyleValue<string> | undefined,
  name: string,
  fallback: Rgba,
  geometryType: GeometryType = 3,
): Evaluator<Rgba> {
  if (value === undefined) return constant(fallback);
  const expr = compile(value, name, COLOR_SPEC);
  if (!expr) return constant(fallback);
  return fromCompiled(expr, geometryType, (v) => {
    const c = v as Color;
    // Color channels are premultiplied by alpha — undo for a plain RGB triple.
    const inv = c.a > 0 ? 1 / c.a : 0;
    return [c.r * inv, c.g * inv, c.b * inv, c.a] as Rgba;
  });
}

/** Compile a numeric property. */
export function compileNumber(
  value: StyleValue<number> | undefined,
  name: string,
  fallback: number,
  geometryType: GeometryType = 3,
): Evaluator<number> {
  if (value === undefined) return constant(fallback);
  const expr = compile(value, name, NUMBER_SPEC);
  if (!expr) return constant(fallback);
  return fromCompiled(expr, geometryType, (v) => v as number);
}

/** Compile a string property (e.g. `model-id`). */
export function compileString(
  value: StyleValue<string> | undefined,
  name: string,
  fallback: string,
  geometryType: GeometryType = 3,
): Evaluator<string> {
  if (value === undefined) return constant(fallback);
  const expr = compile(value, name, STRING_SPEC);
  if (!expr) return constant(fallback);
  return fromCompiled(expr, geometryType, (v) => v as string);
}

/**
 * Compile a 3-number property (rotation / scale / translation).
 *
 * Four spellings are accepted:
 *   `2.15`                        — scalar shorthand for all three components
 *   `[0, 0, 90]`                  — plain constant
 *   `['literal', [0, 0, 90]]`     — or any expression returning a 3-array
 *   `[0, 0, ['get', 'bearing']]`  — per-COMPONENT expressions
 *
 * The last form is the ergonomic one: the style spec has no way to *build* an
 * array from expressions, so a bare array whose elements aren't all numbers is
 * compiled component-wise instead of as a single array expression.
 *
 * Whatever comes back is normalised to a 3-array, so a stray scalar or a
 * mistyped expression degrades to a usable value instead of throwing mid-render.
 */
export function compileVec3(
  value: StyleValue<Vec3> | unknown[] | undefined,
  name: string,
  fallback: Vec3,
  geometryType: GeometryType = 3,
): Evaluator<Vec3> {
  if (value === undefined) return constant(fallback);

  // Scalar shorthand: the same value on all three axes (uniform scaling).
  if (typeof value === 'number') return constant<Vec3>([value, value, value]);

  // A bare 3-array that isn't itself an expression → compile per component, so
  // individual components may be expressions.
  if (Array.isArray(value) && value.length === 3 && !isExpression(value)) {
    if (value.every((v) => typeof v === 'number')) return constant([...value] as Vec3);
    const parts = value.map((v, i) =>
      compileNumber(v as StyleValue<number>, `${name}[${i}]`, fallback[i], geometryType),
    );
    const out: Vec3 = [0, 0, 0];
    return evaluator(
      (zoom, props, id, globalState) => {
        out[0] = parts[0](zoom, props, id, globalState);
        out[1] = parts[1](zoom, props, id, globalState);
        out[2] = parts[2](zoom, props, id, globalState);
        return out;
      },
      parts.every((p) => p.featureConstant),
      new Set(parts.flatMap((p) => [...p.globalStateRefs])),
    );
  }

  const expr = compile(value, name, spec({ ...VEC3_SPEC, default: fallback }));
  if (!expr) return constant(fallback);
  const out: Vec3 = [0, 0, 0];
  return fromCompiled(expr, geometryType, (v) => {
    if (typeof v === 'number') {
      out[0] = out[1] = out[2] = v;
      return out;
    }
    if (Array.isArray(v) && v.length === 3) return v as Vec3;
    return fallback;
  });
}
