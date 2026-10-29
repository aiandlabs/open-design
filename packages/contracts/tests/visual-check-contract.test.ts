import { describe, expect, it } from 'vitest';

import {
  normalizeVisualCheckPrefs,
  normalizeVisualCheckRounds,
  parseVisualCheckRequest,
  VISUAL_CHECK_MAX_CHECKLIST_ITEMS,
  VISUAL_CHECK_MAX_ROUNDS,
  VISUAL_CHECK_VIEWPORT_SIZES,
  visualCheckViewportsFromFlag,
} from '../src/api/visual-check.js';

describe('visual check request contract', () => {
  it('accepts the shared UI and CLI request shape', () => {
    expect(parseVisualCheckRequest({
      fileName: ' index.html ',
      brief: ' pricing page ',
      checklist: [' Hero fits ', '', 'CTA visible'],
      viewports: ['mobile', 'desktop'],
      model: 'zai-org/glm-5.3-flash',
    })).toEqual({
      fileName: 'index.html',
      brief: 'pricing page',
      checklist: ['Hero fits', 'CTA visible'],
      viewports: ['desktop', 'mobile'],
      model: 'zai-org/glm-5.3-flash',
    });
    expect(parseVisualCheckRequest({ fileName: 'a.html' })).toEqual({ fileName: 'a.html' });
  });

  it.each([{ projectId: 'other' }, { apiKey: 'x' }, { outDir: '/tmp' }, { rounds: 3 }])(
    'rejects authority, credential, or path overrides %j',
    (extra) => {
      expect(() => parseVisualCheckRequest({ fileName: 'a.html', ...extra })).toThrow('unsupported');
    },
  );

  it.each([
    {},
    { fileName: '' },
    { fileName: 'a.html', viewports: [] },
    { fileName: 'a.html', viewports: ['tablet'] },
    { fileName: 'a.html', checklist: 'one' },
    { fileName: 'a.html', checklist: [1] },
    { fileName: 'a.html', checklist: Array.from({ length: VISUAL_CHECK_MAX_CHECKLIST_ITEMS + 1 }, (_, i) => `item ${i}`) },
    { fileName: 'a.html', model: '' },
    { fileName: 'a.html', brief: 5 },
  ])('rejects invalid bodies %j', (body) => {
    expect(() => parseVisualCheckRequest(body)).toThrow();
  });

  it('accepts a BCP 47 language and rejects a malformed one', () => {
    expect(parseVisualCheckRequest({ fileName: 'index.html', language: 'ja' }).language).toBe('ja');
    expect(parseVisualCheckRequest({ fileName: 'index.html', language: 'en-GB' }).language).toBe('en-GB');
    expect(() => parseVisualCheckRequest({ fileName: 'index.html', language: 'Japanese please' })).toThrow(/language/);
  });

  it('maps the CLI --viewport flag', () => {
    expect(visualCheckViewportsFromFlag(undefined)).toEqual(['desktop']);
    expect(visualCheckViewportsFromFlag('mobile')).toEqual(['mobile']);
    expect(visualCheckViewportsFromFlag('both')).toEqual(['desktop', 'mobile']);
    expect(() => visualCheckViewportsFromFlag('tablet')).toThrow('--viewport');
  });

  it('has a desktop and a mobile viewport preset', () => {
    expect(VISUAL_CHECK_VIEWPORT_SIZES.desktop.width).toBeGreaterThan(VISUAL_CHECK_VIEWPORT_SIZES.mobile.width);
  });
});

describe('visual check prefs', () => {
  it('clamps rounds to 0..max and defaults to off', () => {
    expect(normalizeVisualCheckRounds(undefined)).toBe(0);
    expect(normalizeVisualCheckRounds(-2)).toBe(0);
    expect(normalizeVisualCheckRounds(2.7)).toBe(2);
    expect(normalizeVisualCheckRounds(99)).toBe(VISUAL_CHECK_MAX_ROUNDS);
    expect(normalizeVisualCheckRounds('2')).toBe(0);
  });

  it('normalizes stored prefs', () => {
    expect(normalizeVisualCheckPrefs({ rounds: 2, model: '  some/model ' })).toEqual({ rounds: 2, model: 'some/model' });
    expect(normalizeVisualCheckPrefs({ rounds: 9, model: '' })).toEqual({ rounds: VISUAL_CHECK_MAX_ROUNDS });
    expect(normalizeVisualCheckPrefs(null)).toBeUndefined();
    expect(normalizeVisualCheckPrefs([1])).toBeUndefined();
  });
});
