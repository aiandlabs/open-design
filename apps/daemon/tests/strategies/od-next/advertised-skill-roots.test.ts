import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { odNextAdvertisedSkillRoots } from '../../../src/strategies/od-next/advertised-skill-roots.js';

const plugins = path.join(path.sep, 'opt', 'od', 'plugins', '_official');
const root = path.join(plugins, 'scenarios/od-next-strategy/assets/task-profiles');

describe('odNextAdvertisedSkillRoots', () => {
  it('allowlists the task-profile root the prompt tells the agent to read', () => {
    const prompt = `Read the matching profile under ${root}/prototype.md before you build.`;
    expect(odNextAdvertisedSkillRoots(prompt, plugins, () => true)).toEqual([root]);
  });

  it('adds nothing when the prompt does not name the root', () => {
    expect(odNextAdvertisedSkillRoots('Build a landing page.', plugins, () => true)).toEqual([]);
    expect(odNextAdvertisedSkillRoots(null, plugins, () => true)).toEqual([]);
  });

  it('adds nothing when the root is missing on disk', () => {
    expect(odNextAdvertisedSkillRoots(`see ${root}`, plugins, () => false)).toEqual([]);
  });
});
