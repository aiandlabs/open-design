// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { FileViewer } from '../../src/components/FileViewer';
import type { ProjectFile } from '../../src/types';

function htmlFile(): ProjectFile {
  return {
    name: 'pages/landing.html',
    path: 'pages/landing.html',
    type: 'file',
    size: 1024,
    mtime: 1710000000,
    kind: 'html',
    mime: 'text/html',
    artifactManifest: {
      version: 1,
      kind: 'html',
      title: 'Landing',
      entry: 'pages/landing.html',
      renderer: 'html',
      exports: ['html'],
    },
  };
}

describe('FileViewer preview toolbar → Check with vision model', () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('opens the visual check dialog for the previewed file', () => {
    render(
      <FileViewer
        projectId="project-1"
        projectKind="prototype"
        file={htmlFile()}
        liveHtml="<html><body><main>Landing</main></body></html>"
      />,
    );
    const button = screen.getByRole('button', { name: 'Check with vision model' });
    fireEvent.click(button);
    const dialog = screen.getByTestId('visual-check-dialog');
    expect(dialog.textContent).toContain('Vision model check');
    expect(dialog.textContent).toContain('pages/landing.html');
  });
});
