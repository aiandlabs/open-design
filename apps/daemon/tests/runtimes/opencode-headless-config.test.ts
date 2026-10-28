import { describe, expect, it } from 'vitest';

import { withOpenCodeHeadlessRunConfig } from '../../src/runtimes/opencode-permissions.js';

describe('withOpenCodeHeadlessRunConfig', () => {
  it('keeps the agent loop running after an auto-rejected permission', () => {
    expect(withOpenCodeHeadlessRunConfig()).toEqual({ experimental: { continue_loop_on_deny: true } });
  });

  it('keeps run-scoped provider config and other experimental keys', () => {
    const provider = { provider: { 'open-design-byok': { name: 'OpenDesign BYOK' } } };
    expect(withOpenCodeHeadlessRunConfig({ ...provider, experimental: { batch_tool: true } })).toEqual({
      ...provider,
      experimental: { batch_tool: true, continue_loop_on_deny: true },
    });
  });
});
