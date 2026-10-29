import { describe, expect, it } from 'vitest';
import { safeMediaOutputName } from '../../src/media/index.js';

describe('media output names', () => {
  it('keeps a safe subfolder, so the file lands where the agent asked', () => {
    expect(safeMediaOutputName('assets/hero-melonpan.png')).toBe('assets/hero-melonpan.png');
  });
  it('still refuses escapes and absolute paths by flattening them', () => {
    expect(safeMediaOutputName('../secret.png')).not.toContain('..');
    expect(safeMediaOutputName('/etc/hero.png').startsWith('/')).toBe(false);
  });
  it('flattens nothing in a plain name', () => {
    expect(safeMediaOutputName('hero.png')).toBe('hero.png');
  });
});
