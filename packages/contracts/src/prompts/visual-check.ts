import { normalizeVisualCheckRounds } from '../api/visual-check.js';

/**
 * The one source of the visual check prompt rule.
 *
 * When Settings → visual check rounds is above 0, every prompt path that can
 * run `od` (legacy classic, legacy slim, OD Next filesystem runs) swaps its
 * "render at most once" rule for this text. At 0 nothing calls this and the
 * prompts are byte-identical to the build without the feature.
 *
 * The wording is deliberately free of the vocabulary that OD Next's
 * planning/Build guard rejects (see `FORBIDDEN_POST_BUILD_SEMANTICS` in
 * `od-next-strategy.ts`), so OD Next carries it as an ordinary guarded
 * instruction block. `packages/contracts/tests/visual-check-prompt.test.ts`
 * holds it to that.
 */
export function renderVisualCheckBudgetRule(rounds: number): string {
  const n = normalizeVisualCheckRounds(rounds);
  if (n <= 0) return '';
  const times = n === 1 ? '1 time' : `${n} times`;
  return [
    `Visual check budget: ${times} per task. The user turned this on in Settings, so it replaces the one-render rule for this run.`,
    `After a substantive visual change to an HTML file, run \`"$OD_NODE_BIN" "$OD_BIN" visual-check <file> --project "$OD_PROJECT_ID" --json\` (add \`--brief "<one-line brief>"\` when the request has specific visual requirements, and \`--viewport both\` when the design must work on phones).`,
    'The command renders the file and a vision model grades the picture. It returns a verdict and, for each requirement, pass or fail, the evidence, and a repair hint.',
    `Run it at most ${times} per task. When items fail, fix the failing items in the file, then run it again only if budget remains. Stop when the verdict is pass or the budget is used.`,
    'Do not launch your own browser or take your own screenshots. If the command itself fails, do not retry it; continue from your static reading of the file.',
  ].join(' ');
}

/** Markdown bullet form used by the legacy charters. */
export function renderVisualCheckBudgetBullet(rounds: number): string {
  const rule = renderVisualCheckBudgetRule(rounds);
  return rule ? `- **Visual check.** ${rule}` : '';
}
