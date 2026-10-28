# Local ComfyUI

Installs a ComfyUI server for the `comfyui` media provider and the `comfyui-agent` asset loop.

```bash
node scripts/comfyui-local/setup.ts --dir <install-dir>            # install
node scripts/comfyui-local/setup.ts --dir <install-dir> --start    # install, then run on 127.0.0.1:8188
```

- ComfyUI is pinned to the commit in `models.json`.
- `--gpu` is `mps` on Apple silicon and `cuda` on Linux by default. Pass `--gpu cpu` to run without a GPU.
- The script installs only models whose licences permit commercial use. It checks each file's sha256 before it moves the file into the models folder.
- The server starts with `--offline`, so ComfyUI's cloud partner nodes are off. The asset loop also refuses any graph that uses a partner node.
- Requires `git` and `uv`.

| Model | File | Licence | Size |
|---|---|---|---|
| `sdxl-base-1.0` (default) | `checkpoints/sd_xl_base_1.0.safetensors` | CreativeML Open RAIL++-M | 6.9 GB |

Stable Diffusion XL needs approximately 8 GB of GPU memory. On an M5 Pro with 24 GB, one 768×768 image at 20 steps takes approximately 30 s, including the model load.
