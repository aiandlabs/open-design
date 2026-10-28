import type { VisualCheckResponse, VisualCheckViewportName } from '@open-design/contracts';
import {
  Button,
  Dialog,
  DialogBody,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  Select,
  Textarea,
} from '@open-design/components';
import { useCallback, useEffect, useId, useRef, useState } from 'react';

import { useT } from '../../i18n';
import { runVisualCheck, visualCheckScreenshotUrl } from '../../providers/visual-check';
import { RemixIcon } from '../RemixIcon';
import styles from './VisualCheckAction.module.css';

type ViewportChoice = 'desktop' | 'mobile' | 'both';

const VIEWPORTS: Record<ViewportChoice, VisualCheckViewportName[]> = {
  desktop: ['desktop'],
  mobile: ['mobile'],
  both: ['desktop', 'mobile'],
};

interface Props {
  projectId: string;
  fileName: string;
  disabled?: boolean;
  disabledTitle?: string;
}

/**
 * File preview toolbar entry point for the visual check. Renders the file
 * through `POST /api/projects/:id/visual-check` (the endpoint
 * `od visual-check` also calls) and shows the vision model's grades.
 */
export function VisualCheckAction({ projectId, fileName, disabled = false, disabledTitle }: Props) {
  const t = useT();
  const titleId = useId();
  const viewportId = useId();
  const briefId = useId();
  const [open, setOpen] = useState(false);
  const [viewport, setViewport] = useState<ViewportChoice>('desktop');
  const [brief, setBrief] = useState('');
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<VisualCheckResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => () => abortRef.current?.abort(), []);
  // A different file starts from a clean panel.
  useEffect(() => {
    setResult(null);
    setError(null);
  }, [fileName]);

  const close = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setRunning(false);
    setOpen(false);
  }, []);

  const run = useCallback(async () => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      const trimmed = brief.trim();
      const response = await runVisualCheck(
        projectId,
        {
          fileName,
          viewports: VIEWPORTS[viewport],
          ...(trimmed ? { brief: trimmed } : {}),
        },
        { signal: controller.signal },
      );
      if (controller.signal.aborted) return;
      setResult(response);
    } catch (err) {
      if (controller.signal.aborted) return;
      setError(t('fileViewer.visualCheckFailed', { message: err instanceof Error ? err.message : String(err) }));
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
        setRunning(false);
      }
    }
  }, [brief, fileName, projectId, t, viewport]);

  const label = t('fileViewer.visualCheck');
  const tooltip = disabled && disabledTitle ? disabledTitle : label;

  return (
    <>
      <button
        type="button"
        className="viewer-action viewer-action-icon od-tooltip"
        data-testid="visual-check-button"
        data-tooltip={tooltip}
        data-tooltip-placement="bottom"
        title={tooltip}
        aria-label={label}
        disabled={disabled}
        onClick={() => setOpen(true)}
      >
        <RemixIcon name="eye-line" size={15} />
      </button>
      {open ? (
        <Dialog
          layout="sectioned"
          className={styles.dialog}
          ariaLabelledBy={titleId}
          onClose={close}
          closeOnEscape
          data-testid="visual-check-dialog"
        >
          <DialogHeader>
            <DialogTitle id={titleId}>{t('fileViewer.visualCheckTitle')}</DialogTitle>
            <DialogDescription>
              {t('fileViewer.visualCheckSubtitle')} <code className={styles.file}>{fileName}</code>
            </DialogDescription>
          </DialogHeader>
          <DialogBody className={styles.body}>
            <div className={styles.form}>
              <label className={styles.label} htmlFor={viewportId}>
                {t('fileViewer.visualCheckViewport')}
              </label>
              <Select
                id={viewportId}
                data-testid="visual-check-viewport"
                value={viewport}
                disabled={running}
                onChange={(event) => setViewport(event.target.value as ViewportChoice)}
              >
                <option value="desktop">{t('fileViewer.visualCheckViewportDesktop')}</option>
                <option value="mobile">{t('fileViewer.visualCheckViewportMobile')}</option>
                <option value="both">{t('fileViewer.visualCheckViewportBoth')}</option>
              </Select>
              <label className={styles.label} htmlFor={briefId}>
                {t('fileViewer.visualCheckBrief')}
              </label>
              <Textarea
                id={briefId}
                data-testid="visual-check-brief"
                rows={2}
                maxLength={8000}
                disabled={running}
                placeholder={t('fileViewer.visualCheckBriefPlaceholder')}
                value={brief}
                onChange={(event) => setBrief(event.target.value)}
              />
            </div>
            {error ? (
              <p className={styles.error} role="alert" data-testid="visual-check-error">
                {error}
              </p>
            ) : null}
            {running ? (
              <p className={styles.status} role="status">
                {t('fileViewer.visualCheckRunning')}
              </p>
            ) : null}
            {result ? <VisualCheckResult projectId={projectId} result={result} /> : null}
          </DialogBody>
          <DialogFooter>
            <Button variant="ghost" onClick={close}>
              {t('common.close')}
            </Button>
            <Button
              variant="primary"
              data-testid="visual-check-run"
              disabled={running}
              onClick={() => void run()}
            >
              {running ? t('fileViewer.visualCheckRunning') : t('fileViewer.visualCheckRun')}
            </Button>
          </DialogFooter>
        </Dialog>
      ) : null}
    </>
  );
}

function VisualCheckResult({ projectId, result }: { projectId: string; result: VisualCheckResponse }) {
  const t = useT();
  const passed = result.verdict === 'pass';
  return (
    <section className={styles.result} data-testid="visual-check-result">
      <div className={styles.verdictRow}>
        <span
          className={`${styles.badge} ${passed ? styles.badgePass : styles.badgeFail}`}
          data-testid="visual-check-verdict"
        >
          {passed ? t('fileViewer.visualCheckPass') : t('fileViewer.visualCheckFail')}
        </span>
        <span className={styles.summary}>{result.summary}</span>
      </div>
      {result.screenshots.length > 0 ? (
        <div className={styles.shots}>
          {result.screenshots.map((shot) => (
            <a
              key={shot.path}
              href={visualCheckScreenshotUrl(projectId, shot.path)}
              target="_blank"
              rel="noreferrer"
              className={styles.shot}
            >
              <img
                src={visualCheckScreenshotUrl(projectId, shot.path)}
                alt={t('fileViewer.visualCheckScreenshotAlt', { viewport: shot.viewport })}
                loading="lazy"
              />
              <span className={styles.shotLabel}>
                {shot.viewport} · {shot.width}×{shot.height}
              </span>
            </a>
          ))}
        </div>
      ) : null}
      <ul className={styles.items}>
        {result.items.map((item) => (
          <li key={item.id} className={styles.item} data-testid="visual-check-item" data-pass={item.pass ? 'true' : 'false'}>
            <div className={styles.itemHead}>
              <span className={`${styles.badge} ${item.pass ? styles.badgePass : styles.badgeFail}`}>
                {item.pass ? t('fileViewer.visualCheckPass') : t('fileViewer.visualCheckFail')}
              </span>
              <span className={styles.itemText}>{item.item}</span>
              {item.viewport ? <span className={styles.itemViewport}>{item.viewport}</span> : null}
            </div>
            {item.evidence ? (
              <p className={styles.detail}>
                <strong>{t('fileViewer.visualCheckEvidence')}:</strong> {item.evidence}
              </p>
            ) : null}
            {!item.pass && item.repairHint ? (
              <p className={styles.detail}>
                <strong>{t('fileViewer.visualCheckRepairHint')}:</strong> {item.repairHint}
              </p>
            ) : null}
          </li>
        ))}
      </ul>
      <p className={styles.meta} data-testid="visual-check-usage">
        {t('fileViewer.visualCheckUsage', {
          model: result.model,
          input: String(result.usage.inputTokens),
          output: String(result.usage.outputTokens),
        })}
      </p>
    </section>
  );
}
