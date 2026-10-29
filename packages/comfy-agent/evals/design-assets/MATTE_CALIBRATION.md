# Matte residue limits

For transparent assets, the loop now renders the subject on a plain backdrop
and cuts it out with a matting model (BiRefNet, through ComfyUI's core
`RemoveBackground` node). It keys a chroma colour in code only when no
matting model is installed. The `matte_residue` check uses three measures from
`matteResidue()` in `src/image.ts`:

| Measure | Meaning | Limit |
|---|---|---|
| Fringe | Share of edge pixels (visible pixels within 2 px of transparency) near the backdrop colour | ≤ 0.10 |
| Backdrop-tinted | Share of all visible pixels near the backdrop colour | ≤ 0.02 |
| Haze | Share of visible pixels that are partly transparent away from the edge band (a shadow, glow or haze) | ≤ 0.05 |

Haze leaves out the soft edge itself. Otherwise a thin subject, such as a
branch or a line drawing, would fail only because it is mostly edge.

## Calibration

* **Data.** These are the 106 rounds of the 13 transparent tasks in the two
  design-asset eval runs: SDXL (run 1) and FLUX.2 [klein] 4B (run 2). They were
  rendered on chroma keys, so they are a hard case for matting. The backdrop is
  green or magenta, and the subjects were not chosen to contrast with it.
* **Masks.** The masks came from `Comfy-Org/BiRefNet` `birefnet.safetensors`
  (sha256 `9ab37426…a154`). The weights were loaded by ComfyUI's own
  `comfy.bg_removal_model` on CPU, in a separate process. The ComfyUI server
  was not used.
* **Procedure.** `calibrate-matte.ts` applies the loop's post-processing to
  each mask (`decontaminateMatte` against the measured backdrop), records the
  measures, and writes contact sheets on white and near-black for labelling.

| Measure | Median | 90th percentile | Maximum |
|---|---|---|---|
| Fringe | 0.0004 | 0.013 | 0.139 |
| Backdrop-tinted | 0 | 0.002 | 0.291 |
| Haze | 0 | 0.0007 | 0.250 |

With the limits above, 5 of the 106 rounds fail. I checked each of these by
eye:

* **Rounds 0, 1 and 2** (SDXL, lantern icon) draw the subject in the key
  colour: a green lantern, a man in a green suit, and green trim. Round 0 also
  has a painted halo, with haze 0.25.
* **Round 37** (klein, map pin) has a key-coloured line along the bottom edge
  of the map. Its fringe is 0.139.
* **Round 92** (klein, raw arm, leaf icon) is a green leaf on a green key, with
  backdrop-tinted 0.025. With matting, the planner chooses a backdrop that
  contrasts with the subject. The raw arm uses white.

The closest clean round is round 18 (klein, leaf icon). Its fringe is 0.066,
and it has no visible fringe at full size. This is why the fringe limit is
0.10, not lower.

On the same rounds, the chroma-key `key_residue` check failed 63 of 106. The
cause is mostly a key tint in soft edges, steam and transparent parts. The
matte has no such tint.

**Status:** these are mattes of renders on a chroma key. The new flow renders
on white, mid-grey or black. Re-check the limits on the first live rounds of
the new flow: the `matting` arm of the eval records them.

## Revision: subject colours close to the backdrop (2026-10-29)

The first live rounds of the new flow showed a false failure. A lavender bell on the mid-grey backdrop (`#808080`) failed with fringe 0.11 and backdrop-tinted 0.132, but its cut-out was clean on black and on magenta. The cause: the bell's own shading is near the backdrop colour, and the check counted those subject pixels as residue. The failure cost one extra round.

`matteResidue()` now takes a baseline from the subject's deep interior, the pixels more than 8 px from transparency, where no backdrop can be left. Fringe is the edge rate above that baseline. Backdrop-tinted counts only the outer 8 px band, above the same baseline. The limits are unchanged.

* The live bell now measures fringe 0.005 and backdrop-tinted 0. Before, it failed.
* The new measures can only be lower than the old ones, so no round that passed before fails now.
* Of the 5 calibration failures, round 37 (an edge line) still fails, and round 0 still fails on haze. Rounds 1, 2 and 92 now pass. Their subject is drawn in the key colour (a green lantern, a man in a green suit, a green leaf on green). That is a content defect for the critic, not leftover backdrop. With matting, the planner picks a backdrop that contrasts with the subject.
* The masks were not kept, so these results come from the reasoning above and the recorded measures, not from a rerun.
