import fs from 'node:fs';
import path from 'node:path';

/**
 * The OD Next strategy prompt names its task-profile directory as an absolute
 * skill root (`{{SKILL_ROOT}}` in `plugins/strategy-package.ts`) and tells the
 * agent to read profiles from it. That directory lives in the bundled plugin
 * folder, outside the project cwd, so it must be on the run's read allowlist —
 * the same invariant `resolveChatExtraAllowedDirs()` keeps for classic skill
 * roots. Without it a headless OpenCode run auto-rejects the read as an
 * `external_directory` access and the turn ends with no deliverable.
 *
 * Returns the root only when the composed prompt actually advertises it and it
 * exists on disk.
 */
export function odNextAdvertisedSkillRoots(
  promptText: string | null | undefined,
  bundledPluginsDir: string,
  existsSync: (p: string) => boolean = fs.existsSync,
): string[] {
  if (!promptText) return [];
  const root = path.join(bundledPluginsDir, 'scenarios/od-next-strategy/assets/task-profiles');
  return promptText.includes(root) && existsSync(root) ? [root] : [];
}
