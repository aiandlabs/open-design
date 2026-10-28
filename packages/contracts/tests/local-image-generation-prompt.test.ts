import { describe, expect, it } from 'vitest';

import {
  LOCAL_MEDIA_PROVIDER_IDS,
  isMediaProviderAllowedByPolicy,
  mediaLocalOnlyDenial,
  normalizeMediaPolicyPrefs,
  resolveMediaLocalOnly,
} from '../src/api/media-local-policy.js';
import { mediaFailureNextStep } from '../src/api/media.js';
import {
  assertOdNextPlanningBuildOnlyV2,
  composeOdNextStrategyRequestPromptV2,
  composeOdNextStrategyStableRequestContextV2,
  type OdNextStrategyRequestRecipeV2,
} from '../src/prompts/od-next-strategy.js';
import {
  renderLocalImageGenerationRule,
  renderLocalImageGenerationSection,
  type LocalImageGenerationPromptInput,
} from '../src/prompts/local-image-generation.js';
import { composeSystemPrompt } from '../src/prompts/system.js';

const recipe: OdNextStrategyRequestRecipeV2 = {
  recipe: 'od-next-plan-build-v2',
  strategyId: 'od-next-strategy',
  strategyVersion: '2.0.0',
  snapshotId: 'snapshot-local-image',
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

const BLOCK = '<od-next-context kind="instruction" name="local-image-generation">';
const INPUTS: ReadonlyArray<LocalImageGenerationPromptInput> = [
  { localOnly: true, comfyuiModel: null },
  { localOnly: true, comfyuiModel: 'comfyui-agent' },
  { localOnly: true, comfyuiModel: 'comfyui-workflow' },
  { localOnly: false, comfyuiModel: 'comfyui-agent' },
  { localOnly: false, comfyuiModel: 'comfyui-workflow' },
];

describe('local-only policy (contracts)', () => {
  it('defaults to on and normalises the stored block', () => {
    expect(resolveMediaLocalOnly(undefined)).toBe(true);
    expect(resolveMediaLocalOnly(null)).toBe(true);
    expect(resolveMediaLocalOnly({ localOnly: false })).toBe(false);
    expect(normalizeMediaPolicyPrefs({ localOnly: 'no' })).toBeUndefined();
    expect(normalizeMediaPolicyPrefs({ localOnly: true, extra: 1 })).toEqual({ localOnly: true });
  });

  it('classifies only ComfyUI, HyperFrames and the stub as local', () => {
    expect([...LOCAL_MEDIA_PROVIDER_IDS]).toEqual(['comfyui', 'hyperframes', 'stub']);
    for (const hosted of ['vela', 'openai', 'nanobanana', 'fal', 'openrouter', 'imagerouter', 'volcengine', 'grok', 'aihubmix', 'senseaudio', 'minimax', 'bfl', 'replicate', 'google', 'leonardo', 'custom-image']) {
      expect(isMediaProviderAllowedByPolicy({ providerId: hosted, surface: 'image', localOnly: true })).toBe(false);
      expect(isMediaProviderAllowedByPolicy({ providerId: hosted, surface: 'video', localOnly: true })).toBe(false);
      expect(isMediaProviderAllowedByPolicy({ providerId: hosted, surface: 'audio', localOnly: true })).toBe(true);
      expect(isMediaProviderAllowedByPolicy({ providerId: hosted, surface: 'image', localOnly: false })).toBe(true);
    }
  });

  it('maps each refusal code to the next step it implies', () => {
    const base = { localOnly: true, providerId: 'vela', model: 'vela/gpt-image-2' };
    const withComfy = mediaLocalOnlyDenial({ ...base, surface: 'image', comfyuiModel: 'comfyui-agent' })!;
    const noComfy = mediaLocalOnlyDenial({ ...base, surface: 'image', comfyuiModel: null })!;
    const video = mediaLocalOnlyDenial({ ...base, surface: 'video', comfyuiModel: 'comfyui-agent' })!;
    expect(mediaFailureNextStep({ code: withComfy.code })).toBe('switch-model');
    expect(mediaFailureNextStep({ code: noComfy.code })).toBe('open-settings');
    expect(mediaFailureNextStep({ code: video.code })).toBe('unsupported');
    expect(noComfy.message).toContain('Settings → Media providers → ComfyUI');
    expect(mediaLocalOnlyDenial({ ...base, surface: 'image', localOnly: false, comfyuiModel: null })).toBeNull();
  });
});

describe('local image generation rule (single source)', () => {
  it('is empty when the policy is off and ComfyUI is not configured', () => {
    expect(renderLocalImageGenerationRule({ localOnly: false, comfyuiModel: null })).toBe('');
    expect(renderLocalImageGenerationRule(undefined)).toBe('');
    expect(renderLocalImageGenerationSection({ localOnly: false, comfyuiModel: null })).toBe('');
  });

  it.each(INPUTS)('passes the OD Next planning/Build guard: %o', (input) => {
    const rule = renderLocalImageGenerationRule(input);
    expect(rule).not.toBe('');
    expect(() => assertOdNextPlanningBuildOnlyV2(rule, 'local-image-generation')).not.toThrow();
  });

  it('names comfyui-agent with --size, --transparent and --negative', () => {
    const rule = renderLocalImageGenerationRule({ localOnly: true, comfyuiModel: 'comfyui-agent' });
    expect(rule).toContain('"$OD_NODE_BIN" "$OD_BIN" media generate --project "$OD_PROJECT_ID" --surface image --model comfyui-agent');
    expect(rule).toContain('--size <W>x<H>');
    expect(rule).toContain('--transparent');
    expect(rule).toContain('--negative');
    expect(rule).toContain('media wait <taskId>');
    expect(rule).toContain('vela/*');
  });

  it('does not offer comfyui-agent flags when only the workflow model can run', () => {
    const rule = renderLocalImageGenerationRule({ localOnly: true, comfyuiModel: 'comfyui-workflow' });
    expect(rule).toContain('--model comfyui-workflow');
    expect(rule).not.toContain('--surface image --model comfyui-agent');
  });

  it('tells the agent there is no image generation when the policy is on without ComfyUI', () => {
    const rule = renderLocalImageGenerationRule({ localOnly: true, comfyuiModel: null });
    expect(rule).toContain('not available in this run');
    expect(rule).toContain('media stock-search');
    expect(rule).toContain('designed placeholder');
    expect(rule).toContain('In the reply, say');
  });
});

describe('OD Next side carries the rule', () => {
  it('adds the instruction block on filesystem runs only', () => {
    const input = { localOnly: true, comfyuiModel: 'comfyui-agent' } as const;
    expect(composeOdNextStrategyStableRequestContextV2({ localImageGeneration: input }, 'filesystem')).toContain(BLOCK);
    expect(composeOdNextStrategyStableRequestContextV2({ localImageGeneration: input }, 'text_artifact')).not.toContain(BLOCK);
    expect(composeOdNextStrategyRequestPromptV2(recipe, { localImageGeneration: input })).toContain('--model comfyui-agent');
  });

  it('is byte-identical to the pre-feature prompt when the policy is off and ComfyUI is not configured', () => {
    const context = { locale: 'en', userInstructions: 'Prefer serif headings.' };
    const baseline = composeOdNextStrategyRequestPromptV2(recipe, context);
    expect(composeOdNextStrategyRequestPromptV2(recipe, {
      ...context,
      localImageGeneration: { localOnly: false, comfyuiModel: null },
    })).toBe(baseline);
    expect(baseline).not.toContain(BLOCK);
  });

  it('reaches OD Next through the API/BYOK fork, and leaves that legacy stack alone', () => {
    const input = { localOnly: true, comfyuiModel: null } as const;
    expect(composeSystemPrompt({ odNextStrategyRecipe: recipe, localImageGeneration: input })).toContain(BLOCK);
    expect(composeSystemPrompt({ odNextStrategyRecipe: recipe, localImageGeneration: { localOnly: false, comfyuiModel: null } })).toBe(
      composeSystemPrompt({ odNextStrategyRecipe: recipe }),
    );
    expect(composeSystemPrompt({ localImageGeneration: input })).toBe(composeSystemPrompt({}));
  });
});
