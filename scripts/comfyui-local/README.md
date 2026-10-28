# Local ComfyUI

Installs a ComfyUI server for the `comfyui` media provider and the `comfyui-agent` asset loop.

```bash
node scripts/comfyui-local/setup.ts --list                                      # sets, sizes, and what fits this machine
node scripts/comfyui-local/setup.ts --dir <install-dir>                         # default sets that fit this machine's memory
node scripts/comfyui-local/setup.ts --dir <install-dir> --sets z-image-turbo,flux2-klein-4b
node scripts/comfyui-local/setup.ts --dir <install-dir> --start                 # install, then run on 127.0.0.1:8188
```

- ComfyUI is pinned to the commit in `models.json`. Every graph uses native ComfyUI nodes; no custom node is needed.
- `--gpu` is `mps` on Apple silicon and `cuda` on Linux by default. Pass `--gpu cpu` to run without a GPU.
- Models come in sets: a diffusion model with its text encoder and VAE. Without `--sets`, the script installs every default set whose memory need fits the machine (`--memory-gb` overrides the detected value). Files shared between sets download once.
- The script installs only models whose licences permit commercial use. It checks each file's sha256 before it moves the file into the models folder.
- The server starts with `--offline`, so ComfyUI's cloud partner nodes are off. The asset loop also refuses any graph that uses a partner node.
- Requires `git` and `uv`.

## Model sets

| Set | Files | Licence | Download | Memory | Use |
|---|---|---|---|---|---|
| `sdxl` (default) | `checkpoints/sd_xl_base_1.0.safetensors` | CreativeML Open RAIL++-M | 6.9 GB | 16 GB | Fallback text-to-image; img2img and inpaint |
| `z-image-turbo` (default) | `diffusion_models/z_image_turbo_int8_convrot.safetensors`, `text_encoders/qwen_3_4b.safetensors`, `vae/ae.safetensors` | Apache-2.0 | 14.6 GB | 24 GB | Text-to-image |
| `flux2-klein-4b` (default) | `diffusion_models/flux-2-klein-4b.safetensors`, `text_encoders/qwen_3_4b.safetensors`, `vae/flux2-vae.safetensors` | Apache-2.0 | 16.1 GB (7.8 GB with Z-Image) | 24 GB | Text-to-image and instruction editing |
| `ming-image-design` | `diffusion_models/ming_image_0.1_design_int8_convrot.safetensors`, `text_encoders/ming_image_0.1_ling_mini_2.0_w4a8.safetensors`, `vae/ming_image_vae_bf16.safetensors` | MIT | 19.2 GB | 32 GB | Text-to-image and instruction editing for UI and poster layouts |

The asset loop uses the first complete set in this order: FLUX.2 [klein] 4B, Z-Image-Turbo, Ming-Image, then SDXL. Instruction edits (`edit_previous`) use FLUX.2 [klein] 4B, else Ming-Image. See `packages/comfy-agent/src/families.ts`.

## Measured on an M5 Pro with 24 GB

Six prompts from the design-asset eval at 1024×1024 on a ComfyUI server that was also running another evaluation, with system swap at 20–40 GB. Times are ComfyUI execution time per image, including model loads; queue waits are excluded.

| Model | Steps | Time per 1024² image | ComfyUI process peak | Checklist result (6 prompts) |
|---|---|---|---|---|
| FLUX.2 [klein] 4B | 4 | 53–63 s | 22–25 GB | 5 of 6 pass; the 404 illustration drew "404" |
| Z-Image-Turbo (int8) | 8 | 83–112 s; 534–1436 s while the machine was swapping | 22–24 GB | 2 of 6 pass; drew a phone UI, a label, "404" and five people instead of three |
| Ming-Image-0.1-Design | 12 | 118 s (one image) | 17 GB | the lantern icon passed; the edit did not finish within an hour |
| SDXL base 1.0 | 24 | 35–205 s | 10–13 GB | 3 of 6 pass |

The peak includes the other evaluation's SDXL model. On an idle machine, Z-Image-Turbo sampled at 8.6 s per step. FLUX.2 [klein] 4B instruction edits ("remove the extra lanterns", "make the background plain white", "change the cup to matcha green") took 67–294 s and kept the rest of the image.

Models that were considered and are not installed, because their licences do not permit commercial use without conditions: Qwen-Image-2.1 (Qwen Research License, non-commercial), FLUX.2 [dev] and FLUX.2 [klein] 9B (FLUX Non-Commercial), FLUX 3 Image Klein (FLUX Kommunity License: free only under US$5M annual revenue), Krea 2 (free only under US$1M annual revenue), Fibo 2.0 (CC BY-NC 4.0), Ideogram 4 (non-commercial), HunyuanImage 2.1 and 3.0 (Tencent Hunyuan Community License: excludes the EU, UK and South Korea). Qwen-Image-2512 and Qwen-Image-Edit-2511 are Apache-2.0, but their 20B transformer does not fit in 24 GB with ComfyUI's native loaders.

## Weights on Apple silicon

- PyTorch's MPS backend cannot hold fp8 tensors, and ComfyUI has no fp8 compute on MPS. Plain fp8 files are converted to bf16 when they load, so they save disk space but not memory. Prefer int8 or bf16 files on a Mac.
- ComfyUI keeps `int8_convrot` and `w4a8` weights quantized on MPS and dequantizes each layer when it runs ("emulated ops"). This halves the memory of the diffusion model compared with bf16.
- ComfyUI 0.43.0 loaded the Qwen3-4B text encoder on the CPU in fp16 (7.7 GB). On unified memory that still counts against the same 24 GB.
