import {
  VISUAL_CHECK_MAX_MODEL_CHARS,
  VISUAL_CHECK_MAX_ROUNDS,
  normalizeVisualCheckRounds,
} from '@open-design/contracts';
import { Input, Select } from '@open-design/components';
import { useId } from 'react';

import { useT } from '../../i18n';
import type { AppConfig } from '../../types';
import styles from './VisualCheckSettings.module.css';

interface Props {
  cfg: AppConfig;
  setCfg: (next: AppConfig) => void;
}

/**
 * Settings → Instructions → Visual check. `rounds` is how many times the agent
 * may run `od visual-check` per task; 0 (the default) leaves prompts unchanged.
 */
export function VisualCheckSettings({ cfg, setCfg }: Props) {
  const t = useT();
  const roundsId = useId();
  const modelId = useId();
  const rounds = normalizeVisualCheckRounds(cfg.visualCheck?.rounds);
  const model = cfg.visualCheck?.model ?? '';

  const update = (patch: { rounds?: number; model?: string }) => {
    const nextRounds = patch.rounds ?? rounds;
    const nextModel = (patch.model ?? model).slice(0, VISUAL_CHECK_MAX_MODEL_CHARS);
    setCfg({
      ...cfg,
      visualCheck: {
        rounds: nextRounds,
        ...(nextModel.trim() ? { model: nextModel } : {}),
      },
    });
  };

  return (
    <div className={styles.card} data-testid="visual-check-settings">
      <div>
        <h4 className={styles.title}>{t('settings.visualCheckTitle')}</h4>
        <p className={styles.hint}>{t('settings.visualCheckDesc')}</p>
      </div>
      <div className={styles.row}>
        <label className={styles.label} htmlFor={roundsId}>
          {t('settings.visualCheckRounds')}
        </label>
        <Select
          id={roundsId}
          className={styles.rounds}
          data-testid="visual-check-rounds"
          value={String(rounds)}
          onChange={(event) => update({ rounds: normalizeVisualCheckRounds(Number(event.target.value)) })}
        >
          {Array.from({ length: VISUAL_CHECK_MAX_ROUNDS + 1 }, (_, n) => (
            <option key={n} value={String(n)}>
              {n === 0 ? t('settings.visualCheckRoundsOff') : String(n)}
            </option>
          ))}
        </Select>
      </div>
      <div className={styles.row}>
        <label className={styles.label} htmlFor={modelId}>
          {t('settings.visualCheckModel')}
        </label>
        <Input
          id={modelId}
          className={styles.model}
          data-testid="visual-check-model"
          type="text"
          spellCheck={false}
          autoComplete="off"
          maxLength={VISUAL_CHECK_MAX_MODEL_CHARS}
          placeholder={t('settings.visualCheckModelAuto')}
          value={model}
          onChange={(event) => update({ model: event.target.value })}
        />
      </div>
      <p className={styles.hint}>{t('settings.visualCheckModelHint')}</p>
    </div>
  );
}
