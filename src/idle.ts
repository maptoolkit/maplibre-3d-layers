/** Resolves once `idle()` holds; looked at once per frame. */
export function whenIdle(idle: () => boolean): Promise<void> {
  return new Promise((resolve) => {
    const check = () => {
      if (idle()) resolve();
      else requestAnimationFrame(check);
    };
    check();
  });
}
