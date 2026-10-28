// Legacy side of the visual check budget (see docs/prompt-composition.md).
//
// Both legacy charters state a one-render budget: the classic charter as the
// `ONE_RENDER_BUDGET_BULLET` and the slim charter as
// `SLIM_ONE_RENDER_BUDGET_RULE`. When Settings → visual check rounds is above
// 0, this swaps whichever one the composed prompt carries for the shared rule
// from `@open-design/contracts` (`renderVisualCheckBudgetRule`), the same text
// OD Next receives as its `visual-check-budget` instruction block. At 0, or on
// a text-artifact run that has no shell to run `od`, the prompt is returned
// untouched, so it is byte-identical to the build without this feature.

import {
  normalizeVisualCheckRounds,
  renderVisualCheckBudgetBullet,
  renderVisualCheckBudgetRule,
  type ExecutionProfile,
} from '@open-design/contracts';

import { SLIM_ONE_RENDER_BUDGET_RULE } from './core-slim.js';
import { ONE_RENDER_BUDGET_BULLET } from './official-system.js';

export function applyVisualCheckBudget(
  prompt: string,
  rounds: number | undefined,
  executionProfile: ExecutionProfile | undefined,
): string {
  const n = normalizeVisualCheckRounds(rounds);
  if (n <= 0 || executionProfile === 'text_artifact') return prompt;
  let next = prompt;
  if (next.includes(ONE_RENDER_BUDGET_BULLET)) {
    next = next.replace(ONE_RENDER_BUDGET_BULLET, renderVisualCheckBudgetBullet(n));
  }
  if (next.includes(SLIM_ONE_RENDER_BUDGET_RULE)) {
    next = next.replace(SLIM_ONE_RENDER_BUDGET_RULE, renderVisualCheckBudgetRule(n));
  }
  return next;
}
