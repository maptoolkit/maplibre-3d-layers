/** Resolves at the first animation frame at which `isIdle()` holds (right away if it holds now). */
export function untilIdle(isIdle: () => boolean): Promise<void> {
  return new Promise((resolve) => {
    const check = () => (isIdle() ? resolve() : requestAnimationFrame(check));
    check();
  });
}
