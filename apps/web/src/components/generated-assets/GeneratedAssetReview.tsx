import type { GeneratedAsset } from '@open-design/contracts';
import { Button } from '@open-design/components';
import { useCallback, useEffect, useState } from 'react';

import { useT } from '../../i18n';
import { listGeneratedAssets, reviewGeneratedAsset } from '../../providers/generated-assets';
import styles from './GeneratedAssetReview.module.css';

interface Props {
  projectId: string;
  fileName: string;
}

/**
 * Image viewer toolbar control for an image the ComfyUI provider generated:
 * shows whether it is accepted or rejected and toggles it through
 * `PUT /api/projects/:id/media/assets/review` (the endpoint `od media review`
 * also calls). Rejected icons are not used as style references for later
 * icons. Renders nothing for other images.
 */
export function GeneratedAssetReview({ projectId, fileName }: Props) {
  const t = useT();
  const [asset, setAsset] = useState<GeneratedAsset | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setAsset(null);
    setError(null);
    listGeneratedAssets(projectId, { signal: controller.signal })
      .then((assets) => setAsset(assets.find((a) => a.path === fileName) ?? null))
      .catch(() => {
        // Not a generated asset, or the daemon is unavailable: show nothing.
      });
    return () => controller.abort();
  }, [projectId, fileName]);

  const toggle = useCallback(async () => {
    if (!asset) return;
    setBusy(true);
    setError(null);
    try {
      setAsset(await reviewGeneratedAsset(projectId, asset.path, asset.review.state === 'rejected' ? 'accepted' : 'rejected'));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }, [asset, projectId]);

  if (!asset) return null;
  const rejected = asset.review.state === 'rejected';
  return (
    <div className={styles.review} title={t('generatedAsset.hint')}>
      <span className={rejected ? styles.rejected : styles.accepted} data-testid="generated-asset-state">
        {rejected ? t('generatedAsset.rejected') : t('generatedAsset.accepted')}
      </span>
      <Button variant="ghost" size="sm" onClick={toggle} disabled={busy}>
        {rejected ? t('generatedAsset.accept') : t('generatedAsset.reject')}
      </Button>
      {error ? (
        <span role="alert" className={styles.error}>
          {t('generatedAsset.error')}
        </span>
      ) : null}
    </div>
  );
}
