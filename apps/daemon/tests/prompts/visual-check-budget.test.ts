import { describe, expect, it } from 'vitest';

import type { OdNextStrategyRequestRecipeV2 } from '@open-design/contracts';

import { renderSlimCoreCharter, SLIM_ONE_RENDER_BUDGET_RULE } from '../../src/prompts/core-slim.js';
import { ONE_RENDER_BUDGET_BULLET, renderOfficialDesignerPrompt } from '../../src/prompts/official-system.js';
import { composeSystemPrompt, type ComposeInput } from '../../src/prompts/system.js';
import {
  applyVisualCheckBudget,
  CLASSIC_VISUAL_CHECK_REWRITES,
  SLIM_VISUAL_CHECK_REWRITES,
} from '../../src/prompts/visual-check-budget.js';

// The exact one-render rules as they read before the visual check feature.
// Pinned as literals so moving them into named constants cannot change a byte.
const CLASSIC_BUDGET_ON_MAIN = '- **Do not loop.** One render check is the budget. Do not spawn a browser, hit a profile/permission/path snag, retry under headless, retry a second binary, then capture desktop + mobile "to be sure." Each such round-trip replays this turn\'s full context into the model and is the single biggest driver of input-token blowup. If the first wrapper render doesn\'t work, move on — a working artifact you reasoned about statically beats three failed screenshot attempts.';
const SLIM_BUDGET_ON_MAIN = '   - Render at most once per task using `"$OD_NODE_BIN" "$OD_BIN" export <file> --project "$OD_PROJECT_ID" --format image --out <output-path>`. Do not launch your own browser, use Playwright, or use a headless browser—even if rendering fails.';

const recipe: OdNextStrategyRequestRecipeV2 = {
  recipe: 'od-next-plan-build-v2',
  strategyId: 'od-next-strategy',
  strategyVersion: '2.0.0',
  snapshotId: 'snapshot-daemon-fork',
  packageHash: 'a'.repeat(64),
  taskProfileDigest: 'b'.repeat(64),
  taskProfileVersion: '2.0.0',
  taskType: 'prototype',
  executionProfile: 'filesystem',
  coreStrategy: '# Core\n\nKeep route and execution facts locked.',
  generalOrchestration: '# Orchestration\n\nPrepare a Design Spec and Full Plan, then Build.',
  taskSkill: '# Prototype\n\nProduce the declared editable prototype.',
  activeStages: [
    { name: 'discovery', atoms: [{ name: 'discovery-question-form' }] },
    { name: 'plan', atoms: [{ name: 'direction-picker' }, { name: 'todo-write' }] },
    { name: 'generate', atoms: [{ name: 'file-write' }, { name: 'live-artifact' }] },
  ],
};

const SCENARIOS: ReadonlyArray<readonly [string, ComposeInput]> = [
  ['classic design', { skillMode: 'prototype', metadata: { kind: 'prototype' } as any }],
  ['slim design', { skillMode: 'prototype', promptCoreVariant: 'slim', metadata: { kind: 'prototype' } as any }],
  ['slim deck', { skillMode: 'deck', promptCoreVariant: 'slim', metadata: { kind: 'deck' } as any }],
  ['classic plain stream', { streamFormat: 'plain', skillMode: 'prototype' }],
  ['slim ask mode', { promptCoreVariant: 'slim', sessionMode: 'chat' as any }],
  ['slim image', { promptCoreVariant: 'slim', skillMode: 'image', metadata: { kind: 'image' } as any }],
];

describe('visual check rounds = 0 leaves every legacy prompt byte-identical', () => {
  it('the extracted constants are the exact pre-feature rule text', () => {
    expect(ONE_RENDER_BUDGET_BULLET).toBe(CLASSIC_BUDGET_ON_MAIN);
    expect(`   - ${SLIM_ONE_RENDER_BUDGET_RULE}`).toBe(SLIM_BUDGET_ON_MAIN);
    expect(renderOfficialDesignerPrompt('filesystem')).toContain(`\n${CLASSIC_BUDGET_ON_MAIN}\n`);
    expect(renderSlimCoreCharter('filesystem')).toContain(`\n${SLIM_BUDGET_ON_MAIN}\n`);
  });

  it.each(SCENARIOS)('%s', (_label, input) => {
    const baseline = composeSystemPrompt(input);
    expect(composeSystemPrompt({ ...input, visualCheckRounds: 0 })).toBe(baseline);
    expect(composeSystemPrompt({ ...input, visualCheckRounds: undefined })).toBe(baseline);
    expect(baseline).not.toContain('Visual check budget');
  });

  it('the OD Next fork is byte-identical at 0', () => {
    expect(composeSystemPrompt({ odNextStrategyRecipe: recipe, visualCheckRounds: 0 })).toBe(
      composeSystemPrompt({ odNextStrategyRecipe: recipe }),
    );
  });
});

describe('visual check rounds > 0 swaps the one-render rule', () => {
  it('classic: replaces the "Do not loop" bullet with the bounded loop', () => {
    const prompt = composeSystemPrompt({ ...SCENARIOS[0]![1], visualCheckRounds: 2 });
    expect(prompt).not.toContain(CLASSIC_BUDGET_ON_MAIN);
    expect(prompt).toContain('- **Visual check.** Visual check budget: 2 times per task.');
    expect(prompt).toContain('visual-check <file> --project "$OD_PROJECT_ID" --json');
    expect(prompt).toContain('fix the failing items');
    expect(prompt).toContain('Stop when the verdict is pass or the budget is used.');
  });

  it('slim: replaces the render-at-most-once rule and keeps the sub-bullet shape', () => {
    const prompt = composeSystemPrompt({ ...SCENARIOS[1]![1], visualCheckRounds: 3 });
    expect(prompt).not.toContain(SLIM_BUDGET_ON_MAIN);
    expect(prompt).toContain('\n   - Visual check budget: 3 times per task.');
    expect(prompt).toContain('Run it at most 3 times per task.');
  });

  it('every neighbouring sentence it rewrites exists verbatim in its charter', () => {
    const classic = renderOfficialDesignerPrompt('filesystem');
    const slim = renderSlimCoreCharter('filesystem');
    for (const [from] of CLASSIC_VISUAL_CHECK_REWRITES) expect(classic).toContain(from);
    for (const [from] of SLIM_VISUAL_CHECK_REWRITES) expect(slim).toContain(from);
  });

  it.each([
    ['classic', 0],
    ['slim', 1],
  ] as const)('%s: no one-render or render-only-when-necessary wording survives', (_label, index) => {
    const prompt = composeSystemPrompt({ ...SCENARIOS[index]![1], visualCheckRounds: 2 });
    for (const stale of [
      'ONE rendered look',
      'verify once before you ship',
      'One render check is the budget',
      'Render at most once per task',
      'Render only when static code review',
      'only when necessary',
      'one-render budget',
    ]) {
      expect(prompt).not.toContain(stale);
    }
  });

  it('only the budget rule and its rewritten neighbours change', () => {
    for (const [index, rewrites, pinned] of [
      [0, CLASSIC_VISUAL_CHECK_REWRITES, CLASSIC_BUDGET_ON_MAIN],
      [1, SLIM_VISUAL_CHECK_REWRITES, SLIM_BUDGET_ON_MAIN],
    ] as const) {
      const input = SCENARIOS[index]![1];
      const before = composeSystemPrompt(input);
      let after = composeSystemPrompt({ ...input, visualCheckRounds: 1 });
      const rule = after.split('\n').find((line) => line.includes('Visual check budget'))!;
      after = after.replace(rule, pinned);
      for (const [from, to] of rewrites) after = after.replace(to, from);
      expect(after).toBe(before);
    }
  });

  it('text-artifact runs and prompts without a design charter are untouched', () => {
    for (const index of [3, 4, 5]) {
      const input = SCENARIOS[index]![1];
      expect(composeSystemPrompt({ ...input, visualCheckRounds: 3 })).toBe(composeSystemPrompt(input));
    }
    expect(applyVisualCheckBudget(CLASSIC_BUDGET_ON_MAIN, 2, 'text_artifact')).toBe(CLASSIC_BUDGET_ON_MAIN);
  });

  it('the OD Next fork carries the same rule as an instruction block', () => {
    const prompt = composeSystemPrompt({ odNextStrategyRecipe: recipe, visualCheckRounds: 2 });
    expect(prompt).toContain('<od-next-context kind="instruction" name="visual-check-budget">');
    expect(prompt).toContain('Run it at most 2 times per task.');
  });
});
