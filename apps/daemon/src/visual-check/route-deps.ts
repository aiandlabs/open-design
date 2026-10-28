// Production wiring for the visual check route: the real renderers, the ai&
// client, the Settings read, and the project-prompt lookup. Kept out of
// server.ts so the server only passes its existing handles in.

import { readFile, rm } from 'node:fs/promises';
import nodePath from 'node:path';

import { AiandClient } from '@open-design/comfy-agent';
import {
  normalizeVisualCheckRounds,
  type VisualCheckPrefs,
} from '@open-design/contracts';

import { findBrowserExecutable, type BrowserSessionService } from '../browser-sessions.js';
import type { DesktopArtifactExporter } from '../server.js';
import { resolveAiandLlmCredentials } from '../media/aiand-credentials.js';
import { buildDesktopArtifactExportInput } from '../pdf-export.js';
import { readProjectFile, resolveProjectDir } from '../projects.js';
import type { RegisterVisualCheckRoutesDeps } from '../routes/visual-check.js';
import type { ProjectPreviewScopeDeps } from '../server-context.js';
import { renderVisualCheckViewports } from './render.js';
import { captureWithSystemBrowser } from './system-browser-capture.js';

const PREVIEW_SCOPE_TTL_MS = 5 * 60 * 1000;

export interface VisualCheckRouteWiring {
  db: unknown;
  getProject: RegisterVisualCheckRoutesDeps['getProject'];
  sendApiError: RegisterVisualCheckRoutesDeps['sendApiError'];
  isLocalSameOrigin: RegisterVisualCheckRoutesDeps['isLocalSameOrigin'];
  resolvedPortRef: { current: number };
  authorizeProjectRequest: RegisterVisualCheckRoutesDeps['authorizeProjectRequest'];
  authorizeProjectToolRequest: RegisterVisualCheckRoutesDeps['authorizeProjectToolRequest'];
  authorizeToolRequest: RegisterVisualCheckRoutesDeps['authorizeToolRequest'];
  /** Media-config root, as passed to resolveProviderConfig elsewhere. */
  PROJECT_ROOT: string;
  PROJECTS_DIR: string;
  readVisualCheckPrefs: () => Promise<VisualCheckPrefs | null | undefined>;
  listConversations: (db: any, projectId: string) => Array<{ id: string; createdAt?: number }>;
  listMessages: (db: any, conversationId: string) => Array<{ role?: string; content?: unknown }>;
  projectPreviewScopes: Pick<ProjectPreviewScopeDeps, 'mint' | 'revoke'>;
  daemonUrlRef: { current: string };
  desktopArtifactExporter?: DesktopArtifactExporter | null;
  browserSessions: Pick<BrowserSessionService, 'create' | 'close'>;
}

function previewUrl(daemonUrl: string, projectId: string, scope: string, fileName: string): { baseHref: string; url: string } {
  const clean = fileName.replace(/^\/+/, '');
  const root = `${daemonUrl.replace(/\/+$/, '')}/api/projects/${encodeURIComponent(projectId)}/preview/${encodeURIComponent(scope)}/`;
  const dir = nodePath.posix.dirname(clean);
  const encodedDir = !dir || dir === '.' ? '' : `${dir.split('/').filter(Boolean).map(encodeURIComponent).join('/')}/`;
  return {
    baseHref: `${root}${encodedDir}`,
    url: `${root}${clean.split('/').filter(Boolean).map(encodeURIComponent).join('/')}`,
  };
}

/** The first user message of the oldest conversation, or the pending prompt. */
export function readProjectOriginalPrompt(
  project: { pendingPrompt?: string | null } | null | undefined,
  conversations: Array<{ id: string; createdAt?: number }>,
  listMessages: (conversationId: string) => Array<{ role?: string; content?: unknown }>,
): string | null {
  const pending = typeof project?.pendingPrompt === 'string' ? project.pendingPrompt.trim() : '';
  if (pending) return pending;
  const ordered = [...conversations].sort((a, b) => (a.createdAt ?? 0) - (b.createdAt ?? 0));
  for (const conversation of ordered) {
    const first = listMessages(conversation.id).find((m) => m.role === 'user' && typeof m.content === 'string' && m.content.trim());
    if (first) return String(first.content).trim();
  }
  return null;
}

export function createVisualCheckRouteDeps(w: VisualCheckRouteWiring): RegisterVisualCheckRoutesDeps {
  return {
    db: w.db,
    getProject: w.getProject,
    sendApiError: w.sendApiError,
    isLocalSameOrigin: w.isLocalSameOrigin,
    resolvedPortRef: w.resolvedPortRef,
    authorizeProjectRequest: w.authorizeProjectRequest,
    authorizeProjectToolRequest: w.authorizeProjectToolRequest,
    authorizeToolRequest: w.authorizeToolRequest,
    projectDir: (projectId, metadata) => resolveProjectDir(w.PROJECTS_DIR, projectId, metadata as any),
    assertProjectFile: async (projectId, fileName, metadata) => {
      await readProjectFile(w.PROJECTS_DIR, projectId, fileName, metadata as any);
    },
    readRounds: async () => normalizeVisualCheckRounds((await w.readVisualCheckPrefs())?.rounds),
    serviceDeps: ({ projectId, metadata }) => ({
      readPrefs: w.readVisualCheckPrefs,
      readProjectPrompt: async () => readProjectOriginalPrompt(
        w.getProject(w.db, projectId) as { pendingPrompt?: string | null } | null,
        w.listConversations(w.db, projectId),
        (conversationId) => w.listMessages(w.db, conversationId),
      ),
      createClient: async () => {
        const creds = await resolveAiandLlmCredentials(w.PROJECT_ROOT);
        if (!creds.apiKey) return null;
        return new AiandClient({ apiKey: creds.apiKey, baseUrl: creds.baseUrl, shape: creds.shape });
      },
      render: async ({ fileName, viewports }) => {
        const scope = w.projectPreviewScopes.mint(projectId, null, { ttlMs: PREVIEW_SCOPE_TTL_MS });
        try {
          const { baseHref, url } = previewUrl(w.daemonUrlRef.current, projectId, scope, fileName);
          const exporter = w.desktopArtifactExporter;
          return await renderVisualCheckViewports(viewports, {
            renderWithDesktop: typeof exporter === 'function'
              ? async ({ width, height }) => {
                  const input = await buildDesktopArtifactExportInput({
                    baseHref,
                    daemonUrl: w.daemonUrlRef.current,
                    deck: false,
                    fileName,
                    format: 'image',
                    imageFormat: 'png',
                    metadata: metadata as any,
                    projectId,
                    projectsRoot: w.PROJECTS_DIR,
                    width,
                    height,
                  });
                  const result = await exporter(input);
                  if (!result.ok || typeof result.path !== 'string') {
                    throw new Error(result.error || 'desktop renderer returned no image');
                  }
                  try {
                    return await readFile(result.path);
                  } finally {
                    await rm(result.path, { force: true }).catch(() => {});
                  }
                }
              : null,
            renderWithSystemBrowser: findBrowserExecutable()
              ? (input) => captureWithSystemBrowser(w.browserSessions, { url, ...input })
              : null,
          });
        } finally {
          w.projectPreviewScopes.revoke(scope);
        }
      },
    }),
  };
}
