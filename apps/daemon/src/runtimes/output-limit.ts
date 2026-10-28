import { runSideEffectsForRun, type RunSideEffectLedger } from './run-lifecycle-analytics.js';

export const OUTPUT_LIMIT_NO_DELIVERABLE_MESSAGE =
  'The model reached its output-token limit before it wrote the design, so nothing was saved. ' +
  'Reply "continue" to resume, or choose a model with a larger output limit.';

/**
 * A clean exit is not a delivery when the model was cut off at its
 * output-token limit and nothing reached the project.
 *
 * Reasoning models can spend most of the output budget thinking, so the step
 * that carries the `write` call is the one that gets truncated; the runtime
 * drops the half-written call and still exits 0. Holds when the turn was
 * truncated (`run.truncatedMidTurn`, folded from a `max_tokens` stop reason)
 * and neither the project's files (`artifactCount`, from the filesystem diff
 * when the run has a project) nor the tool stream show a written artifact or
 * live artifact. A truncated run that did write its file stays succeeded and
 * is reported as unfinished instead.
 */
export function runEndedOnOutputLimitWithoutDeliverable(
  run: {
    truncatedMidTurn?: boolean;
    sideEffectLedger?: RunSideEffectLedger;
    events?: unknown;
  },
  artifactCount: number | null | undefined,
): boolean {
  if (run.truncatedMidTurn !== true) return false;
  if (typeof artifactCount === 'number' && artifactCount > 0) return false;
  const sideEffects = runSideEffectsForRun(run);
  return !sideEffects.artifactWriteSeen && !sideEffects.liveArtifactSeen;
}
