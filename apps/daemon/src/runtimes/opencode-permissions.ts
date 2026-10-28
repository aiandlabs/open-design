import { agentCapabilities } from './capabilities.js';
import type { RuntimeAgentDef } from './types.js';

export const OPENCODE_SKIP_PERMISSIONS_FLAG = '--dangerously-skip-permissions';
export const OPENCODE_WORKSPACE_DIR_FLAG = '--dir';

export const OPENCODE_PERMISSION_CAPABILITY = {
  helpArgs: ['run', '--help'],
  capabilityFlags: {
    [OPENCODE_SKIP_PERMISSIONS_FLAG]: 'skipPermissions',
  },
} satisfies Pick<RuntimeAgentDef, 'helpArgs' | 'capabilityFlags'>;

export function appendOpenCodePermissionBypass(args: string[], agentId: string): void {
  if (agentCapabilities.get(agentId)?.skipPermissions) {
    args.push(OPENCODE_SKIP_PERMISSIONS_FLAG);
  }
}

/**
 * Pin OpenCode's workspace to the resolved project directory.
 *
 * OpenCode does not treat its process cwd as the project: it walks up to the
 * nearest enclosing git root and adopts THAT as the worktree (verified with
 * `opencode debug scrap`, whose every registered project is a git root). A
 * managed project directory is not a git repository, and a development install
 * keeps the daemon data directory under the repository root — so OpenCode walks
 * past the project and adopts the whole Open Design checkout.
 *
 * The consequences are all silent: the agent names the repository root as its
 * workspace and writes the deliverable there, the project directory stays
 * empty, `snapshotProjectArtifactsAsync(cwd)` sees nothing, and the Run reports
 * `no_artifact`. `permission.external_directory` cannot catch it either — once
 * the repository is the worktree, writing inside it is an in-project write.
 *
 * Sent unconditionally, matching the BYOK OpenCode definition that already
 * carried this workaround. Gating it on the `--help` capability probe would
 * turn a missing flag back into the silent data-loss bug; an OpenCode build
 * without `--dir` should fail loudly at spawn instead.
 */
export function appendOpenCodeWorkspaceDir(
  args: string[],
  cwd: string | null | undefined,
): void {
  if (typeof cwd !== 'string' || cwd.length === 0) return;
  args.push(OPENCODE_WORKSPACE_DIR_FLAG, cwd);
}

/**
 * Run-scoped OpenCode config that every headless run carries.
 *
 * A headless `opencode run` cannot ask the user, so it auto-rejects any
 * permission the daemon did not grant up front (an `external_directory` read
 * the prompt suggested, or a path the model mistyped). The OpenCode companion
 * that byok-opencode resolves ends the whole agent loop on a rejection unless
 * `experimental.continue_loop_on_deny` is set: the run exits 0 right after
 * that tool call, mid-plan, with no deliverable. With the flag, the rejection
 * is an ordinary tool error and the model carries on without that action.
 * Newer OpenCode builds continue on their own and accept the key.
 *
 * Merged under any run-scoped provider config; an `experimental` block the
 * caller passes keeps its other keys.
 */
export function withOpenCodeHeadlessRunConfig(
  extraConfig: Record<string, unknown> = {},
): Record<string, unknown> {
  const experimental =
    extraConfig.experimental && typeof extraConfig.experimental === 'object' && !Array.isArray(extraConfig.experimental)
      ? extraConfig.experimental as Record<string, unknown>
      : {};
  return {
    ...extraConfig,
    experimental: { ...experimental, continue_loop_on_deny: true },
  };
}
