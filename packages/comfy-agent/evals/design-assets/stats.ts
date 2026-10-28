// Small, dependency-free statistics for paired arm comparisons and judge agreement.

export function mean(xs: number[]): number | null {
  return xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : null;
}

export function median(xs: number[]): number | null {
  if (!xs.length) return null;
  const s = xs.slice().sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2;
}

/** Linear-interpolated quantile, q in [0, 1]. */
export function quantile(xs: number[], q: number): number | null {
  if (!xs.length) return null;
  const s = xs.slice().sort((a, b) => a - b);
  const pos = (s.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return s[lo]! + (s[hi]! - s[lo]!) * (pos - lo);
}

export type Agreement = {
  n: number;
  agree: number;
  /** Observed agreement. */
  po: number | null;
  /** Chance agreement from each rater's marginal pass rate. */
  pe: number | null;
  /** Cohen's kappa; null when it is undefined (no ratings, or both raters constant and identical). */
  kappa: number | null;
  passRateA: number | null;
  passRateB: number | null;
};

/** Cohen's kappa for two raters giving binary grades to the same items. */
export function cohensKappa(a: boolean[], b: boolean[]): Agreement {
  if (a.length !== b.length) throw new Error("cohensKappa: rating arrays differ in length");
  const n = a.length;
  if (n === 0) return { n: 0, agree: 0, po: null, pe: null, kappa: null, passRateA: null, passRateB: null };
  let agree = 0;
  let passA = 0;
  let passB = 0;
  for (let i = 0; i < n; i += 1) {
    if (a[i] === b[i]) agree += 1;
    if (a[i]) passA += 1;
    if (b[i]) passB += 1;
  }
  const po = agree / n;
  const pa = passA / n;
  const pb = passB / n;
  const pe = pa * pb + (1 - pa) * (1 - pb);
  const kappa = pe >= 1 ? null : (po - pe) / (1 - pe);
  return { n, agree, po, pe, kappa, passRateA: pa, passRateB: pb };
}

/** log of n choose k. */
function logChoose(n: number, k: number): number {
  let s = 0;
  for (let i = 1; i <= k; i += 1) s += Math.log(n - k + i) - Math.log(i);
  return s;
}

/**
 * Exact two-sided sign test: under H0 improvements and regressions are
 * equally likely; ties are dropped. Returns p = 1 when there are no untied pairs.
 */
export function signTest(improved: number, regressed: number): number {
  const n = improved + regressed;
  if (n === 0) return 1;
  const k = Math.min(improved, regressed);
  let tail = 0;
  for (let i = 0; i <= k; i += 1) tail += Math.exp(logChoose(n, i) - n * Math.LN2);
  return Math.min(1, 2 * tail);
}

/** Deterministic PRNG so bootstrap intervals are reproducible. */
export function mulberry32(seed: number): () => number {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

/** Percentile bootstrap CI for the mean of paired differences. */
export function bootstrapMeanCi(diffs: number[], options: { iterations?: number; level?: number; seed?: number } = {}): { low: number; high: number } | null {
  if (!diffs.length) return null;
  const iterations = options.iterations ?? 10_000;
  const level = options.level ?? 0.95;
  const rand = mulberry32(options.seed ?? 12345);
  const means: number[] = new Array(iterations);
  for (let it = 0; it < iterations; it += 1) {
    let s = 0;
    for (let i = 0; i < diffs.length; i += 1) s += diffs[Math.floor(rand() * diffs.length)]!;
    means[it] = s / diffs.length;
  }
  return { low: quantile(means, (1 - level) / 2)!, high: quantile(means, 1 - (1 - level) / 2)! };
}

export type PairedComparison = {
  n: number;
  improved: number;
  tied: number;
  regressed: number;
  meanDiff: number | null;
  ci95: { low: number; high: number } | null;
  signTestP: number;
};

/** Compare paired per-task values (b − a). Differences within `eps` count as ties. */
export function comparePaired(pairs: { a: number; b: number }[], options: { eps?: number; seed?: number } = {}): PairedComparison {
  const eps = options.eps ?? 1e-9;
  const diffs = pairs.map((p) => p.b - p.a);
  const improved = diffs.filter((d) => d > eps).length;
  const regressed = diffs.filter((d) => d < -eps).length;
  return {
    n: diffs.length,
    improved,
    tied: diffs.length - improved - regressed,
    regressed,
    meanDiff: mean(diffs),
    ci95: bootstrapMeanCi(diffs, options.seed === undefined ? {} : { seed: options.seed }),
    signTestP: signTest(improved, regressed),
  };
}
