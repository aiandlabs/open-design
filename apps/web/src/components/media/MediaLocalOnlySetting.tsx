import {
  isLocalMediaProvider,
  resolveMediaLocalOnly,
} from '@open-design/contracts';

import { useT } from '../../i18n';
import { IMAGE_MODELS, VIDEO_MODELS } from '../../media/models';
import type { AppConfig } from '../../types';
import styles from './MediaLocalOnlySetting.module.css';

const IMAGE_OR_VIDEO_PROVIDERS: ReadonlySet<string> = new Set(
  [...IMAGE_MODELS, ...VIDEO_MODELS].map((model) => model.provider),
);

/** Whether "Local image generation only" is in effect for this config (absent → on). */
export function isMediaLocalOnly(cfg: Pick<AppConfig, 'media'>): boolean {
  return resolveMediaLocalOnly(cfg.media);
}

/**
 * Whether the policy keeps this provider from generating images or video.
 * Audio-only providers and the local ComfyUI / HyperFrames providers are never
 * blocked.
 */
export function isProviderBlockedByLocalOnly(
  cfg: Pick<AppConfig, 'media'>,
  providerId: string,
): boolean {
  return isMediaLocalOnly(cfg)
    && IMAGE_OR_VIDEO_PROVIDERS.has(providerId)
    && !isLocalMediaProvider(providerId);
}

interface Props {
  cfg: AppConfig;
  setCfg: (update: (current: AppConfig) => AppConfig) => void;
}

/**
 * Settings → Media providers → "Local image generation only". On (the
 * default) means the daemon refuses hosted image and video providers and the
 * local ComfyUI server is the only image route. Writes app-config
 * `media.localOnly`, the same key `od media policy --local-only` sets.
 */
export function MediaLocalOnlySetting({ cfg, setCfg }: Props) {
  const t = useT();
  const enabled = isMediaLocalOnly(cfg);
  return (
    <div className={styles.wrap} data-testid="media-local-only-setting">
      <button
        type="button"
        className={`toggle-row${enabled ? ' on' : ''}`}
        role="switch"
        aria-checked={enabled}
        data-testid="media-local-only-toggle"
        onClick={() => setCfg((current) => ({
          ...current,
          media: { localOnly: !isMediaLocalOnly(current) },
        }))}
      >
        <span className="toggle-row-text">
          <span className="toggle-row-label">{t('settings.mediaLocalOnly')}</span>
          <span className="toggle-row-hint">{t('settings.mediaLocalOnlyHint')}</span>
        </span>
        <span className="toggle-row-switch" aria-hidden="true" />
      </button>
    </div>
  );
}

/** Dims a hosted image/video provider's pill while the policy is on. */
export const mediaLocalOnlyBlockedChipClass = styles.blockedChip ?? '';

/** The note shown on a hosted image/video provider while the policy is on. */
export function MediaLocalOnlyBlockedNote() {
  const t = useT();
  return (
    <p className={styles.blockedNote} role="note" data-testid="media-local-only-blocked-note">
      {t('settings.mediaLocalOnlyBlocked')}
    </p>
  );
}
