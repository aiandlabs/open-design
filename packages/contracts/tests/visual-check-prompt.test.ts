import { describe, expect, it } from 'vitest';

import {
  assertOdNextPlanningBuildOnlyV2,
  composeOdNextStrategyRequestPromptV2,
  composeOdNextStrategyStableRequestContextV2,
  type OdNextStrategyRequestRecipeV2,
} from '../src/prompts/od-next-strategy.js';
import { composeSystemPrompt } from '../src/prompts/system.js';
import {
  renderVisualCheckBudgetBullet,
  renderVisualCheckBudgetRule,
} from '../src/prompts/visual-check.js';

const recipe: OdNextStrategyRequestRecipeV2 = {
  recipe: 'od-next-plan-build-v2',
  strategyId: 'od-next-strategy',
  strategyVersion: '2.0.0',
  snapshotId: 'snapshot-visual-check',
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

const BLOCK = '<od-next-context kind="instruction" name="visual-check-budget">';

describe('visual check budget rule (single source)', () => {
  it('is empty at 0 and states the budget above 0', () => {
    expect(renderVisualCheckBudgetRule(0)).toBe('');
    expect(renderVisualCheckBudgetBullet(0)).toBe('');
    const rule = renderVisualCheckBudgetRule(2);
    expect(rule).toContain('Visual check budget: 2 times per task.');
    expect(rule).toContain('visual-check <file> --project "$OD_PROJECT_ID" --json');
    expect(rule).toContain('Run it at most 2 times per task.');
    expect(rule).toContain('fix the failing items');
    expect(rule).toContain('Stop when the verdict is pass or the budget is used.');
    expect(renderVisualCheckBudgetRule(1)).toContain('1 time per task');
    expect(renderVisualCheckBudgetRule(50)).toContain('3 times per task');
  });

  it.each([1, 2, 3])('passes the OD Next planning/Build guard at %i rounds', (rounds) => {
    expect(() => assertOdNextPlanningBuildOnlyV2(renderVisualCheckBudgetRule(rounds), 'visual-check-budget')).not.toThrow();
  });
});

describe('OD Next side carries the visual check budget', () => {
  it('adds the instruction block only when rounds is above 0 on a filesystem run', () => {
    const on = composeOdNextStrategyStableRequestContextV2({ visualCheckRounds: 2 }, 'filesystem');
    expect(on).toContain(BLOCK);
    expect(on).toContain('Run it at most 2 times per task.');
    expect(composeOdNextStrategyStableRequestContextV2({ visualCheckRounds: 2 }, 'text_artifact')).not.toContain(BLOCK);
    expect(composeOdNextStrategyRequestPromptV2(recipe, { visualCheckRounds: 1 })).toContain(BLOCK);
  });

  it('is byte-identical to the pre-feature prompt when rounds is 0 or absent', () => {
    const context = { locale: 'en', userInstructions: 'Prefer serif headings.' };
    const baseline = composeOdNextStrategyRequestPromptV2(recipe, context);
    expect(composeOdNextStrategyRequestPromptV2(recipe, { ...context, visualCheckRounds: 0 })).toBe(baseline);
    expect(baseline).not.toContain(BLOCK);
    expect(baseline).not.toContain('Visual check budget');
    expect(composeOdNextStrategyStableRequestContextV2({ ...context, visualCheckRounds: 0 })).toBe(
      composeOdNextStrategyStableRequestContextV2(context),
    );
  });

  it('reaches OD Next through the API/BYOK fork as well', () => {
    expect(composeSystemPrompt({ odNextStrategyRecipe: recipe, visualCheckRounds: 3 })).toContain(BLOCK);
    expect(composeSystemPrompt({ odNextStrategyRecipe: recipe, visualCheckRounds: 0 })).toBe(
      composeSystemPrompt({ odNextStrategyRecipe: recipe }),
    );
  });

  it('leaves the API/BYOK legacy stack untouched (no shell, no budget rule to swap)', () => {
    expect(composeSystemPrompt({ visualCheckRounds: 3 })).toBe(composeSystemPrompt({}));
  });
});
