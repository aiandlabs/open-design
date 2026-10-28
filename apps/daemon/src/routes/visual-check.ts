// POST /api/projects/:id/visual-check — render one project file and grade it
// with an ai& vision model. Shared by the file preview's "Check with vision
// model" action and `od visual-check` (which an agent runs with its run-scoped
// tool token when Settings → visual check rounds is above 0).
//
// GET /api/projects/:id/visual-check/screenshot?path=… — serve a screenshot
// the POST saved under the project's `.visual-check/` folder.

import { readFile } from 'node:fs/promises';
import path from 'node:path';

import type { Express } from 'express';
import {
  normalizeVisualCheckRounds,
  parseVisualCheckRequest,
} from '@open-design/contracts';
import { ModelSelectionError } from '@open-design/comfy-agent';

import type {
  AuthorizeProjectRequest,
  AuthorizeProjectToolRequest,
} from '../collab/project-request-authority.js';
import { VISUAL_CHECK_TOOL_ENDPOINT } from '../tool-tokens.js';
import { VisualCheckRenderError } from '../visual-check/render.js';
import {
  isVisualCheckScreenshotPath,
  runVisualCheck,
  VisualCheckConfigError,
  type VisualCheckServiceDeps,
} from '../visual-check/service.js';

export const VISUAL_CHECK_TOOL_OPERATION = 'project:visual-check';

const RENDERABLE_FILE_RE = /\.html?$/i;
const MAX_TRACKED_RUNS = 500;

export interface RegisterVisualCheckRoutesDeps {
  db: unknown;
  getProject: (db: any, id: string) => { id: string; metadata?: any } | null | undefined;
  sendApiError: (res: any, status: number, code: string, message: string) => any;
  isLocalSameOrigin: (req: any, port: number) => boolean;
  resolvedPortRef: { current: number };
  authorizeProjectRequest: AuthorizeProjectRequest;
  authorizeProjectToolRequest: AuthorizeProjectToolRequest;
  authorizeToolRequest: (
    req: any,
    res: any,
    operation: string,
    options: { endpoint: string },
  ) => { projectId: string; runId: string } | null;
  /** Absolute project folder; imported-folder projects resolve to their baseDir. */
  projectDir: (projectId: string, metadata: unknown) => string;
  /** Throws (ENOENT) when the file is missing from the project. */
  assertProjectFile: (projectId: string, fileName: string, metadata: unknown) => Promise<void>;
  /** Settings → visual check rounds, read per request. */
  readRounds: () => Promise<number>;
  /** Everything runVisualCheck needs except the project folder. */
  serviceDeps: (input: { projectId: string; metadata: unknown }) => Omit<VisualCheckServiceDeps, 'projectDir'>;
}

/**
 * A run may call visual check at most `rounds` times (Settings), or once when
 * rounds is 0 — the same single render the charter allows without the
 * feature. Browser/CLI callers without a tool token are not counted: they are
 * the user asking directly.
 */
export function createVisualCheckRunBudget() {
  const used = new Map<string, number>();
  return {
    tryConsume(runId: string, rounds: number): { ok: true; used: number; budget: number } | { ok: false; used: number; budget: number } {
      const budget = Math.max(1, normalizeVisualCheckRounds(rounds));
      const count = used.get(runId) ?? 0;
      if (count >= budget) return { ok: false, used: count, budget };
      used.delete(runId);
      used.set(runId, count + 1);
      while (used.size > MAX_TRACKED_RUNS) {
        const oldest = used.keys().next().value;
        if (oldest === undefined) break;
        used.delete(oldest);
      }
      return { ok: true, used: count + 1, budget };
    },
    refund(runId: string) {
      const count = used.get(runId);
      if (count && count > 0) used.set(runId, count - 1);
    },
  };
}

export function registerVisualCheckRoutes(app: Express, ctx: RegisterVisualCheckRoutesDeps): void {
  const { db, getProject, sendApiError } = ctx;
  const budget = createVisualCheckRunBudget();

  app.post('/api/projects/:id/visual-check', async (req, res) => {
    if (!ctx.isLocalSameOrigin(req, ctx.resolvedPortRef.current)) {
      return sendApiError(res, 403, 'FORBIDDEN', 'cross-origin request rejected');
    }
    const project = getProject(db, req.params.id);
    if (!project) return sendApiError(res, 404, 'PROJECT_NOT_FOUND', 'project not found');

    const bearer = /^Bearer(?:\s|$)/i.test(req.get('authorization')?.trim() ?? '');
    const grant = bearer
      ? ctx.authorizeToolRequest(req, res, VISUAL_CHECK_TOOL_OPERATION, { endpoint: VISUAL_CHECK_TOOL_ENDPOINT })
      : null;
    if (bearer && !grant) return;
    if (grant && grant.projectId !== project.id) {
      return sendApiError(res, 403, 'FORBIDDEN', 'tool token belongs to a different project');
    }
    if (grant) {
      if (!await ctx.authorizeProjectToolRequest(res, grant.projectId, { mode: 'write', capability: 'writeFiles' })) return;
    } else if (!await ctx.authorizeProjectRequest(req, res, project.id, { mode: 'write', capability: 'writeFiles' })) {
      return;
    }

    let request;
    try {
      request = parseVisualCheckRequest(req.body);
    } catch (err) {
      return sendApiError(res, 400, 'BAD_REQUEST', err instanceof Error ? err.message : String(err));
    }
    if (!RENDERABLE_FILE_RE.test(request.fileName)) {
      return sendApiError(res, 400, 'BAD_REQUEST', 'visual check renders .html files only');
    }
    try {
      await ctx.assertProjectFile(project.id, request.fileName, project.metadata);
    } catch (err: any) {
      const missing = err?.code === 'ENOENT';
      return sendApiError(
        res,
        missing ? 404 : 400,
        missing ? 'FILE_NOT_FOUND' : 'BAD_REQUEST',
        missing ? `file not found: ${request.fileName}` : String(err?.message ?? err),
      );
    }

    if (grant) {
      const rounds = await ctx.readRounds().catch(() => 0);
      const consumed = budget.tryConsume(grant.runId, rounds);
      if (!consumed.ok) {
        return sendApiError(
          res,
          429,
          'VISUAL_CHECK_BUDGET_EXHAUSTED',
          `visual check budget for this task is used (${consumed.used}/${consumed.budget}); deliver the current file`,
        );
      }
    }

    try {
      const response = await runVisualCheck(request, {
        ...ctx.serviceDeps({ projectId: project.id, metadata: project.metadata }),
        projectDir: ctx.projectDir(project.id, project.metadata),
      });
      return res.json(response);
    } catch (err: any) {
      // A failure that never reached the model does not spend the run's budget.
      const modelRejected = err instanceof ModelSelectionError;
      if (grant && (err instanceof VisualCheckRenderError || err instanceof VisualCheckConfigError || modelRejected)) {
        budget.refund(grant.runId);
      }
      if (err instanceof VisualCheckConfigError) {
        return sendApiError(res, 400, err.code, err.message);
      }
      if (modelRejected) {
        return sendApiError(res, 400, 'VISUAL_CHECK_MODEL_INVALID', err.message);
      }
      if (err instanceof VisualCheckRenderError) {
        return sendApiError(res, 503, err.code, err.message);
      }
      const status = typeof err?.status === 'number' && err.status >= 400 && err.status < 600 ? 502 : 500;
      const code = typeof err?.code === 'string' ? err.code : 'VISUAL_CHECK_FAILED';
      return sendApiError(res, status, code, String(err?.message ?? err));
    }
  });

  app.get('/api/projects/:id/visual-check/screenshot', async (req, res) => {
    const project = getProject(db, req.params.id);
    if (!project) return sendApiError(res, 404, 'PROJECT_NOT_FOUND', 'project not found');
    if (!await ctx.authorizeProjectRequest(req, res, project.id, { mode: 'read' })) return;
    const rel = typeof req.query.path === 'string' ? req.query.path : '';
    if (!isVisualCheckScreenshotPath(rel)) {
      return sendApiError(res, 400, 'BAD_REQUEST', 'path must be a .visual-check screenshot path');
    }
    const root = path.resolve(ctx.projectDir(project.id, project.metadata));
    const abs = path.resolve(root, ...rel.split('/'));
    if (!abs.startsWith(root + path.sep)) {
      return sendApiError(res, 400, 'BAD_REQUEST', 'path escapes the project');
    }
    try {
      const bytes = await readFile(abs);
      res.setHeader('Content-Type', 'image/png');
      res.setHeader('Cache-Control', 'private, max-age=31536000, immutable');
      return res.send(bytes);
    } catch {
      return sendApiError(res, 404, 'FILE_NOT_FOUND', 'screenshot not found');
    }
  });
}
