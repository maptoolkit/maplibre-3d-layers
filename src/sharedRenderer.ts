import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';

interface Shared {
  renderer: THREE.WebGLRenderer;
  /** How many layers hold it. */
  users: number;
  environment?: THREE.Texture;
}

const shared = new WeakMap<WebGL2RenderingContext, Shared>();

/**
 * The three.js renderer for a map's WebGL context: one for all layers of this
 * library on that map. A renderer of its own per layer compiled the same
 * shader programs once per layer — the buildings and the landmarks, or two
 * layers of trees, draw with the same ones — and that is paid when a scene
 * starts, on the main thread. Every layer resets the renderer's state before
 * it draws (MapLibre draws in between), so sharing it changes nothing else.
 * Give it back with {@link releaseRenderer}.
 */
export function acquireRenderer(canvas: HTMLCanvasElement, gl: WebGL2RenderingContext): THREE.WebGLRenderer {
  let s = shared.get(gl);
  if (!s) {
    const renderer = new THREE.WebGLRenderer({ canvas, context: gl, antialias: true });
    renderer.autoClear = false;
    shared.set(gl, (s = { renderer, users: 0 }));
  }
  s.users++;
  return s.renderer;
}

/** A layer is done with the context's renderer; the last one disposes of it. */
export function releaseRenderer(gl: WebGL2RenderingContext) {
  const s = shared.get(gl);
  if (!s || --s.users > 0) return;
  s.environment?.dispose();
  s.renderer.dispose();
  shared.delete(gl);
}

/**
 * The light environment of glTF materials (a neutral room), prefiltered once
 * per renderer and shared by its layers: how strongly it lights is the scene's
 * own (`environmentIntensity`). Leaves the renderer's state reset.
 */
export function sharedEnvironment(gl: WebGL2RenderingContext): THREE.Texture | undefined {
  const s = shared.get(gl);
  if (!s) return undefined;
  if (!s.environment) {
    const pmrem = new THREE.PMREMGenerator(s.renderer);
    const room = new RoomEnvironment();
    s.environment = pmrem.fromScene(room, 0.04).texture;
    room.dispose();
    pmrem.dispose();
    s.renderer.resetState(); // PMREM rendered into the shared context
  }
  return s.environment;
}
