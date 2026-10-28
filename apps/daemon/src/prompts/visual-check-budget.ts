// Legacy side of the visual check budget (see docs/prompt-composition.md).
//
// Both legacy charters state a one-render budget: the classic charter as the
// `ONE_RENDER_BUDGET_BULLET` and the slim charter as
// `SLIM_ONE_RENDER_BUDGET_RULE`. When Settings → visual check rounds is above
// 0, this swaps whichever one the composed prompt carries for the shared rule
// from `@open-design/contracts` (`renderVisualCheckBudgetRule`), the same text
// OD Next receives as its `visual-check-budget` instruction block, and
// rewrites the few neighbouring sentences that would otherwise still say
// "render once / only when necessary", so the prompt gives one instruction.
// At 0, or on a text-artifact run that has no shell to run `od`, the prompt is
// returned untouched, so it is byte-identical to the build without the
// feature.
//
// The neighbouring sentences are matched as exact text. If a charter edit
// changes one of them, `tests/prompts/visual-check-budget.test.ts` fails
// (it asserts every target below is present in the charter it belongs to).

import {
  normalizeVisualCheckRounds,
  renderVisualCheckBudgetBullet,
  renderVisualCheckBudgetRule,
  type ExecutionProfile,
} from '@open-design/contracts';

import { SLIM_ONE_RENDER_BUDGET_RULE } from './core-slim.js';
import { ONE_RENDER_BUDGET_BULLET } from './official-system.js';

/** Classic-charter sentences that contradict a multi-round visual check. */
export const CLASSIC_VISUAL_CHECK_REWRITES: ReadonlyArray<readonly [string, string]> = [
  [
    '## Verification — converge at the end, in one pass\nVerification is a single deliberate step at the END of the turn, not a running activity you interleave with building. Build the whole thing first; verify once before you ship.',
    '## Verification — converge at the end, within the visual check budget\nVerification happens at the END of the turn, not as a running activity you interleave with building. Build the whole thing first, then check it with the visual check below.',
  ],
  [
    '- **Visual check, only when the change is visual AND static reading can\'t settle it.** Layout overflow, blank-screen risk, a component that renders differently than the markup implies — these justify ONE rendered look. When you need it, route through the OpenDesign tool wrappers (`"$OD_NODE_BIN" "$OD_BIN" tools ...`), which render in the unsandboxed daemon. Do NOT launch your own browser to do this.',
    '- **Rendered check.** Do not render the page any other way, and do NOT launch your own browser. Every rendered check goes through the visual check in the next bullet, which renders in the unsandboxed daemon.',
  ],
];

/** Slim-charter sentences that contradict a multi-round visual check. */
export const SLIM_VISUAL_CHECK_REWRITES: ReadonlyArray<readonly [string, string]> = [
  [
    '5. **Inspect the rendered result only when necessary:**\n   - Render only when static code review cannot determine whether the layout overflows, elements collide, or similar visual issues are present.',
    '5. **Inspect the rendered result with the visual check:**\n   - After a substantive visual change, check the rendered result with the visual check below instead of relying on static review alone.',
  ],
  [
    'and it does not count against the one-render budget.',
    'and it does not count against the visual check budget.',
  ],
];

function rewriteAll(prompt: string, rewrites: ReadonlyArray<readonly [string, string]>): string {
  let next = prompt;
  for (const [from, to] of rewrites) next = next.replace(from, to);
  return next;
}

export function applyVisualCheckBudget(
  prompt: string,
  rounds: number | undefined,
  executionProfile: ExecutionProfile | undefined,
): string {
  const n = normalizeVisualCheckRounds(rounds);
  if (n <= 0 || executionProfile === 'text_artifact') return prompt;
  let next = prompt;
  if (next.includes(ONE_RENDER_BUDGET_BULLET)) {
    next = rewriteAll(next, CLASSIC_VISUAL_CHECK_REWRITES)
      .replace(ONE_RENDER_BUDGET_BULLET, renderVisualCheckBudgetBullet(n));
  }
  if (next.includes(SLIM_ONE_RENDER_BUDGET_RULE)) {
    next = rewriteAll(next, SLIM_VISUAL_CHECK_REWRITES)
      .replace(SLIM_ONE_RENDER_BUDGET_RULE, renderVisualCheckBudgetRule(n));
  }
  return next;
}
