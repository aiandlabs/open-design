// Render one project file to a PNG per viewport for visual check.
//
// Order of preference:
//   1. The desktop app's bundled Chromium, through the same
//      `desktopArtifactExporter` + `buildDesktopArtifactExportInput` pair that
//      `od export --format image` uses. Only reachable while the desktop app
//      is running.
//   2. The system Chrome / Edge / Chromium (`browser-sessions.ts`), driven
//      over CDP. This covers headless daemons and `tools-dev run web`.
// Both load the file through a short-lived project preview scope, so
// relative assets resolve exactly as in the preview pane.

import {
  VISUAL_CHECK_VIEWPORT_SIZES,
  type VisualCheckRenderer,
  type VisualCheckViewportName,
} from '@open-design/contracts';

export interface VisualCheckRenderedImage {
  viewport: VisualCheckViewportName;
  png: Buffer;
  width: number;
  height: number;
}

export interface VisualCheckRenderResult {
  renderer: VisualCheckRenderer;
  images: VisualCheckRenderedImage[];
}

export interface VisualCheckRenderDeps {
  /** Desktop path. Absent outside the desktop runtime. */
  renderWithDesktop?: ((input: { viewport: VisualCheckViewportName; width: number; height: number }) => Promise<Buffer>) | null;
  /** System-browser path. Absent when no browser executable is installed. */
  renderWithSystemBrowser?: ((input: { viewport: VisualCheckViewportName; width: number; height: number; mobile: boolean }) => Promise<Buffer>) | null;
}

export class VisualCheckRenderError extends Error {
  readonly code = 'VISUAL_CHECK_RENDER_UNAVAILABLE';
}

/** Width/height from a PNG IHDR chunk; null when the bytes are not a PNG. */
export function pngDimensions(png: Buffer): { width: number; height: number } | null {
  if (png.length < 24 || png.readUInt32BE(0) !== 0x89504e47 || png.toString('ascii', 12, 16) !== 'IHDR') return null;
  return { width: png.readUInt32BE(16), height: png.readUInt32BE(20) };
}

export async function renderVisualCheckViewports(
  viewports: readonly VisualCheckViewportName[],
  deps: VisualCheckRenderDeps,
): Promise<VisualCheckRenderResult> {
  const failures: string[] = [];
  const attempt = async (
    renderer: VisualCheckRenderer,
    fn: (viewport: VisualCheckViewportName) => Promise<Buffer>,
  ): Promise<VisualCheckRenderResult | null> => {
    try {
      const images: VisualCheckRenderedImage[] = [];
      for (const viewport of viewports) {
        const png = await fn(viewport);
        const dims = pngDimensions(png);
        if (!dims) throw new Error(`${renderer} renderer returned a non-PNG image`);
        images.push({ viewport, png, ...dims });
      }
      return { renderer, images };
    } catch (err) {
      failures.push(`${renderer}: ${err instanceof Error ? err.message : String(err)}`);
      return null;
    }
  };
  if (deps.renderWithDesktop) {
    const desktop = deps.renderWithDesktop;
    const result = await attempt('desktop', (viewport) =>
      desktop({ viewport, ...VISUAL_CHECK_VIEWPORT_SIZES[viewport] }));
    if (result) return result;
  }
  if (deps.renderWithSystemBrowser) {
    const browser = deps.renderWithSystemBrowser;
    const result = await attempt('system-browser', (viewport) =>
      browser({ viewport, ...VISUAL_CHECK_VIEWPORT_SIZES[viewport], mobile: viewport === 'mobile' }));
    if (result) return result;
  } else {
    failures.push('system-browser: no Chrome, Edge, or Chromium executable found (set OD_BROWSER_EXECUTABLE_PATH)');
  }
  throw new VisualCheckRenderError(`could not render the file: ${failures.join('; ')}`);
}
