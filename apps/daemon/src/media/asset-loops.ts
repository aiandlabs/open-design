// Asset loops this daemon process is running, by absolute work directory. A
// loop's progress trace stays marked running if the daemon stops mid-loop;
// the generating list asks here so such a trace is not shown as live.

const running = new Set<string>();

/** Track a loop until its promise settles; returns the same promise. */
export function trackAssetLoop<T>(workDir: string, loop: Promise<T>): Promise<T> {
  running.add(workDir);
  return loop.finally(() => running.delete(workDir));
}

export function assetLoopRunning(workDir: string): boolean {
  return running.has(workDir);
}
