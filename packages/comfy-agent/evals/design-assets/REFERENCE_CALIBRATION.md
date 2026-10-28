# Near-copy limits for reference images

A shape reference (an emoji or a Material Symbol) gives the image model the
shape and concept of an icon. The output must be a new drawing, not a copy of
the reference. `referenceSimilarity()` in `src/image.ts` compares an output
with each reference it used. The `reference_copy` check fails when both of
these are true:

* **Silhouette IoU ≥ 0.8.** Each subject is cut to its bounding box and scaled
  to 64 × 64 px first, so moving or resizing a copy does not hide it.
* **Edge structure ≥ 0.7.** This is the mean SSIM of the two colour-edge maps:
  the largest per-channel Sobel magnitude, lightly blurred. Edges stay where
  the parts of a drawing meet, so a copy in a new palette is still found.

A matching silhouette alone is expected, because that is what a shape
reference is for. A near-copy also keeps the reference's internal structure.

## Calibration

Script: `pnpm --filter @open-design/comfy-agent exec tsx evals/design-assets/calibrate-near-copy.ts --cache <dir>`.
Run on 2026-10-29 with 60 common object, food, nature and animal emoji.

* **Redraws.** These are the same emoji drawn by different artists: Apple
  Color Emoji (system font), Fluent Emoji 3D and Noto Emoji. They stand in for
  an acceptable output: the same concept, drawn anew. There are 178 pairs.
* **Copies.** These are one emoji scaled (60–90 %), moved, recoloured (channels
  rotated and lightened) and softened by a down/up resample. One variant out of
  four is also flattened onto white. There are 716 copies.

| Measure | Redraws: median | Redraws: 95th percentile | Redraws: maximum | Copies: 5th percentile | Copies: median |
|---|---|---|---|---|---|
| Silhouette IoU | 0.78 | 0.93 | 0.97 | 0.84 | 0.98 |
| Edge structure | 0.24 | 0.57 | 0.64 | 0.78 | 0.97 |

Silhouette IoU alone does not separate the two groups. Edge structure does.

| Limits | Redraws flagged | Copies caught |
|---|---|---|
| IoU ≥ 0.8, structure ≥ 0.7 (chosen) | 0 / 178 | 681 / 716 (95 %) |
| IoU ≥ 0.85, structure ≥ 0.7 | 0 / 178 | 671 / 716 |
| IoU ≥ 0.9, structure ≥ 0.75 | 0 / 178 | 648 / 716 |

Of the 35 copies that are missed, 26 were flattened onto white. For an opaque
image, the subject is found by its difference from the border colour. A pale
subject on white is then only partly found. Transparent icons, which are the
case this check is for, do not have this problem.

**Status:** these are synthetic copies. They are not outputs that the image
model actually copied. When live runs produce flagged outputs, review them and
re-check these limits.
