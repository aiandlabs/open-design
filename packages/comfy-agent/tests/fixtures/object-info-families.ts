import type { ObjectInfo } from "../../src/types.js";

// Node definitions copied verbatim from a live ComfyUI 0.43.0 server's
// `/object_info` (commit 79cf64c, Apple silicon, with the Z-Image-Turbo,
// FLUX.2 [klein] 4B and Ming-Image-0.1-Design files installed). Only the
// classes the split-loader and edit graphs use are kept. The loader file
// lists are the real installed names.

export const FAMILY_NODES: ObjectInfo = {
  "UNETLoader": {
    "input": {
      "required": {
        "unet_name": [
          [
            "flux-2-klein-4b.safetensors",
            "ming_image_0.1_design_int8_convrot.safetensors",
            "z_image_turbo_int8_convrot.safetensors"
          ]
        ],
        "weight_dtype": [
          [
            "default",
            "fp8_e4m3fn",
            "fp8_e4m3fn_fast",
            "fp8_e5m2"
          ],
          {
            "advanced": true
          }
        ]
      }
    },
    "input_order": {
      "required": [
        "unet_name",
        "weight_dtype"
      ]
    },
    "is_input_list": false,
    "output": [
      "MODEL"
    ],
    "output_is_list": [
      false
    ],
    "output_name": [
      "MODEL"
    ],
    "name": "UNETLoader",
    "display_name": "Load Diffusion Model",
    "description": "",
    "python_module": "nodes",
    "category": "model/loaders",
    "output_node": false,
    "has_intermediate_output": false,
    "search_aliases": []
  },
  "CLIPLoader": {
    "input": {
      "required": {
        "clip_name": [
          [
            "ming_image_0.1_ling_mini_2.0_w4a8.safetensors",
            "qwen_3_4b.safetensors"
          ]
        ],
        "type": [
          [
            "stable_diffusion",
            "stable_cascade",
            "sd3",
            "stable_audio",
            "mochi",
            "ltxv",
            "pixart",
            "cosmos",
            "lumina2",
            "wan",
            "hidream",
            "chroma",
            "ace",
            "omnigen2",
            "qwen_image",
            "hunyuan_image",
            "flux2",
            "ovis",
            "longcat_image",
            "cogvideox",
            "lens",
            "pixeldit",
            "ideogram4",
            "boogu",
            "krea2",
            "joyimage",
            "mage",
            "minimax",
            "yue2",
            "flux3",
            "moss_image",
            "fibo2"
          ]
        ]
      },
      "optional": {
        "device": [
          [
            "default",
            "cpu"
          ],
          {
            "advanced": true
          }
        ]
      }
    },
    "input_order": {
      "required": [
        "clip_name",
        "type"
      ],
      "optional": [
        "device"
      ]
    },
    "is_input_list": false,
    "output": [
      "CLIP"
    ],
    "output_is_list": [
      false
    ],
    "output_name": [
      "CLIP"
    ],
    "name": "CLIPLoader",
    "display_name": "Load CLIP",
    "description": "Recipes:\nsd: clip-l\nstable cascade: clip-g\nsd3: t5 xxl / clip-g / clip-l\nstable audio: t5 base\nmochi: t5 xxl\ncogvideox: t5 xxl (226-token padding)\ncosmos: old t5 xxl\nlumina2: gemma 2 2B\nwan: umt5 xxl\nhidream: llama-3.1 (Recommend) or t5\nomnigen2: qwen vl 2.5 3B\njoyimage: qwen3-vl 8B\nlens: gpt-oss-20b\npixeldit: gemma 2 2B elm\nminimax: MiniMax H3 Qwen3-VL or Music3 Qwen/RVQ\nflux3: qwen3-vl 4B\nfibo2: qwen3-vl 4B",
    "python_module": "nodes",
    "category": "model/loaders",
    "output_node": false,
    "has_intermediate_output": false,
    "search_aliases": []
  },
  "VAELoader": {
    "input": {
      "required": {
        "vae_name": [
          [
            "ae.safetensors",
            "flux2-vae.safetensors",
            "ming_image_vae_bf16.safetensors",
            "pixel_space"
          ]
        ]
      }
    },
    "input_order": {
      "required": [
        "vae_name"
      ]
    },
    "is_input_list": false,
    "output": [
      "VAE"
    ],
    "output_is_list": [
      false
    ],
    "output_name": [
      "VAE"
    ],
    "name": "VAELoader",
    "display_name": "Load VAE",
    "description": "",
    "python_module": "nodes",
    "category": "model/loaders",
    "output_node": false,
    "has_intermediate_output": false,
    "search_aliases": []
  },
  "LoraLoaderModelOnly": {
    "input": {
      "required": {
        "model": [
          "MODEL"
        ],
        "lora_name": [
          []
        ],
        "strength_model": [
          "FLOAT",
          {
            "default": 1.0,
            "min": -100.0,
            "max": 100.0,
            "step": 0.01
          }
        ]
      }
    },
    "input_order": {
      "required": [
        "model",
        "lora_name",
        "strength_model"
      ]
    },
    "is_input_list": false,
    "output": [
      "MODEL"
    ],
    "output_is_list": [
      false
    ],
    "output_name": [
      "MODEL"
    ],
    "name": "LoraLoaderModelOnly",
    "display_name": "Load LoRA",
    "description": "This LoRAs loader is used to modify the diffusion model, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together.",
    "python_module": "nodes",
    "category": "model/loaders",
    "output_node": false,
    "has_intermediate_output": false,
    "output_tooltips": [
      "The modified diffusion model.",
      "The modified CLIP model."
    ],
    "search_aliases": [
      "lora",
      "load lora",
      "apply lora",
      "lora loader",
      "lora model"
    ],
    "essentials_category": "Image Generation"
  },
  "ModelSamplingAuraFlow": {
    "input": {
      "required": {
        "model": [
          "MODEL"
        ],
        "shift": [
          "FLOAT",
          {
            "default": 1.73,
            "min": 0.0,
            "max": 100.0,
            "step": 0.01
          }
        ]
      },
      "optional": {
        "sampling": [
          [
            "flow",
            "img_to_img_velocity"
          ],
          {
            "default": "flow",
            "advanced": true
          }
        ]
      }
    },
    "input_order": {
      "required": [
        "model",
        "shift"
      ],
      "optional": [
        "sampling"
      ]
    },
    "is_input_list": false,
    "output": [
      "MODEL"
    ],
    "output_is_list": [
      false
    ],
    "output_name": [
      "MODEL"
    ],
    "name": "ModelSamplingAuraFlow",
    "display_name": "ModelSamplingAuraFlow",
    "description": "",
    "python_module": "comfy_extras.nodes_model_advanced",
    "category": "model/patch",
    "output_node": false,
    "has_intermediate_output": false,
    "search_aliases": []
  },
  "ModelSamplingFlux": {
    "input": {
      "required": {
        "model": [
          "MODEL"
        ],
        "max_shift": [
          "FLOAT",
          {
            "default": 1.15,
            "min": 0.0,
            "max": 100.0,
            "step": 0.01,
            "advanced": true
          }
        ],
        "base_shift": [
          "FLOAT",
          {
            "default": 0.5,
            "min": 0.0,
            "max": 100.0,
            "step": 0.01,
            "advanced": true
          }
        ],
        "width": [
          "INT",
          {
            "default": 1024,
            "min": 16,
            "max": 16384,
            "step": 8
          }
        ],
        "height": [
          "INT",
          {
            "default": 1024,
            "min": 16,
            "max": 16384,
            "step": 8
          }
        ]
      }
    },
    "input_order": {
      "required": [
        "model",
        "max_shift",
        "base_shift",
        "width",
        "height"
      ]
    },
    "is_input_list": false,
    "output": [
      "MODEL"
    ],
    "output_is_list": [
      false
    ],
    "output_name": [
      "MODEL"
    ],
    "name": "ModelSamplingFlux",
    "display_name": "ModelSamplingFlux",
    "description": "",
    "python_module": "comfy_extras.nodes_model_advanced",
    "category": "model/patch/flux",
    "output_node": false,
    "has_intermediate_output": false,
    "search_aliases": []
  },
  "EmptySD3LatentImage": {
    "input": {
      "required": {
        "width": [
          "INT",
          {
            "default": 1024,
            "min": 16,
            "max": 16384,
            "step": 16
          }
        ],
        "height": [
          "INT",
          {
            "default": 1024,
            "min": 16,
            "max": 16384,
            "step": 16
          }
        ],
        "batch_size": [
          "INT",
          {
            "default": 1,
            "min": 1,
            "max": 4096
          }
        ]
      }
    },
    "input_order": {
      "required": [
        "width",
        "height",
        "batch_size"
      ]
    },
    "is_input_list": false,
    "output": [
      "LATENT"
    ],
    "output_is_list": [
      false
    ],
    "output_name": [
      "LATENT"
    ],
    "output_tooltips": [
      null
    ],
    "output_matchtypes": null,
    "name": "EmptySD3LatentImage",
    "display_name": null,
    "description": "",
    "python_module": "comfy_extras.nodes_sd3",
    "category": "model/latent/stable diffusion",
    "output_node": false,
    "deprecated": false,
    "experimental": false,
    "dev_only": false,
    "api_node": false,
    "price_badge": null,
    "search_aliases": null,
    "essentials_category": null,
    "has_intermediate_output": false
  },
  "EmptyFlux2LatentImage": {
    "input": {
      "required": {
        "width": [
          "INT",
          {
            "default": 1024,
            "min": 16,
            "max": 16384,
            "step": 16
          }
        ],
        "height": [
          "INT",
          {
            "default": 1024,
            "min": 16,
            "max": 16384,
            "step": 16
          }
        ],
        "batch_size": [
          "INT",
          {
            "default": 1,
            "min": 1,
            "max": 4096
          }
        ]
      }
    },
    "input_order": {
      "required": [
        "width",
        "height",
        "batch_size"
      ]
    },
    "is_input_list": false,
    "output": [
      "LATENT"
    ],
    "output_is_list": [
      false
    ],
    "output_name": [
      "LATENT"
    ],
    "output_tooltips": [
      null
    ],
    "output_matchtypes": null,
    "name": "EmptyFlux2LatentImage",
    "display_name": "Empty Flux 2 Latent",
    "description": "",
    "python_module": "comfy_extras.nodes_flux",
    "category": "model/latent/flux",
    "output_node": false,
    "deprecated": false,
    "experimental": false,
    "dev_only": false,
    "api_node": false,
    "price_badge": null,
    "search_aliases": null,
    "essentials_category": null,
    "has_intermediate_output": false
  },
  "ConditioningZeroOut": {
    "input": {
      "required": {
        "conditioning": [
          "CONDITIONING"
        ]
      }
    },
    "input_order": {
      "required": [
        "conditioning"
      ]
    },
    "is_input_list": false,
    "output": [
      "CONDITIONING"
    ],
    "output_is_list": [
      false
    ],
    "output_name": [
      "CONDITIONING"
    ],
    "name": "ConditioningZeroOut",
    "display_name": "Conditioning Zero Out",
    "description": "",
    "python_module": "nodes",
    "category": "model/conditioning/transform",
    "output_node": false,
    "has_intermediate_output": false,
    "search_aliases": [
      "null conditioning",
      "clear conditioning"
    ]
  },
  "KSamplerSelect": {
    "input": {
      "required": {
        "sampler_name": [
          "COMBO",
          {
            "multiselect": false,
            "options": [
              "euler",
              "euler_cfg_pp",
              "euler_ancestral",
              "euler_ancestral_cfg_pp",
              "heun",
              "heunpp2",
              "exp_heun_2_x0",
              "exp_heun_2_x0_sde",
              "dpm_2",
              "dpm_2_ancestral",
              "lms",
              "dpm_fast",
              "dpm_adaptive",
              "dpmpp_2s_ancestral",
              "dpmpp_2s_ancestral_cfg_pp",
              "dpmpp_sde",
              "dpmpp_sde_gpu",
              "dpmpp_2m",
              "dpmpp_2m_cfg_pp",
              "dpmpp_2m_sde",
              "dpmpp_2m_sde_gpu",
              "dpmpp_2m_sde_heun",
              "dpmpp_2m_sde_heun_gpu",
              "dpmpp_3m_sde",
              "dpmpp_3m_sde_gpu",
              "ddpm",
              "lcm",
              "ipndm",
              "ipndm_v",
              "deis",
              "cfgpp_ud10_ab",
              "res_multistep",
              "res_multistep_cfg_pp",
              "res_multistep_ancestral",
              "res_multistep_ancestral_cfg_pp",
              "gradient_estimation",
              "gradient_estimation_cfg_pp",
              "er_sde",
              "seeds_2",
              "seeds_3",
              "sa_solver",
              "sa_solver_pece",
              "ddim",
              "uni_pc",
              "uni_pc_bh2"
            ]
          }
        ]
      }
    },
    "input_order": {
      "required": [
        "sampler_name"
      ]
    },
    "is_input_list": false,
    "output": [
      "SAMPLER"
    ],
    "output_is_list": [
      false
    ],
    "output_name": [
      "SAMPLER"
    ],
    "output_tooltips": [
      null
    ],
    "output_matchtypes": null,
    "name": "KSamplerSelect",
    "display_name": null,
    "description": "",
    "python_module": "comfy_extras.nodes_custom_sampler",
    "category": "model/sampling/samplers",
    "output_node": false,
    "deprecated": false,
    "experimental": false,
    "dev_only": false,
    "api_node": false,
    "price_badge": null,
    "search_aliases": null,
    "essentials_category": null,
    "has_intermediate_output": false
  },
  "RandomNoise": {
    "input": {
      "required": {
        "noise_seed": [
          "INT",
          {
            "default": 0,
            "min": 0,
            "max": 18446744073709551615,
            "control_after_generate": true
          }
        ]
      }
    },
    "input_order": {
      "required": [
        "noise_seed"
      ]
    },
    "is_input_list": false,
    "output": [
      "NOISE"
    ],
    "output_is_list": [
      false
    ],
    "output_name": [
      "NOISE"
    ],
    "output_tooltips": [
      null
    ],
    "output_matchtypes": null,
    "name": "RandomNoise",
    "display_name": null,
    "description": "",
    "python_module": "comfy_extras.nodes_custom_sampler",
    "category": "model/sampling/noise",
    "output_node": false,
    "deprecated": false,
    "experimental": false,
    "dev_only": false,
    "api_node": false,
    "price_badge": null,
    "search_aliases": null,
    "essentials_category": null,
    "has_intermediate_output": false
  },
  "CFGGuider": {
    "input": {
      "required": {
        "model": [
          "MODEL",
          {}
        ],
        "positive": [
          "CONDITIONING",
          {}
        ],
        "negative": [
          "CONDITIONING",
          {}
        ],
        "cfg": [
          "FLOAT",
          {
            "default": 8.0,
            "min": 0.0,
            "max": 100.0,
            "step": 0.1,
            "round": 0.01
          }
        ]
      }
    },
    "input_order": {
      "required": [
        "model",
        "positive",
        "negative",
        "cfg"
      ]
    },
    "is_input_list": false,
    "output": [
      "GUIDER"
    ],
    "output_is_list": [
      false
    ],
    "output_name": [
      "GUIDER"
    ],
    "output_tooltips": [
      null
    ],
    "output_matchtypes": null,
    "name": "CFGGuider",
    "display_name": "CFG Guider",
    "description": "",
    "python_module": "comfy_extras.nodes_custom_sampler",
    "category": "model/sampling/guiders",
    "output_node": false,
    "deprecated": false,
    "experimental": false,
    "dev_only": false,
    "api_node": false,
    "price_badge": null,
    "search_aliases": null,
    "essentials_category": null,
    "has_intermediate_output": false
  },
  "BasicGuider": {
    "input": {
      "required": {
        "model": [
          "MODEL",
          {}
        ],
        "conditioning": [
          "CONDITIONING",
          {}
        ]
      }
    },
    "input_order": {
      "required": [
        "model",
        "conditioning"
      ]
    },
    "is_input_list": false,
    "output": [
      "GUIDER"
    ],
    "output_is_list": [
      false
    ],
    "output_name": [
      "GUIDER"
    ],
    "output_tooltips": [
      null
    ],
    "output_matchtypes": null,
    "name": "BasicGuider",
    "display_name": "Basic Guider",
    "description": "",
    "python_module": "comfy_extras.nodes_custom_sampler",
    "category": "model/sampling/guiders",
    "output_node": false,
    "deprecated": false,
    "experimental": false,
    "dev_only": false,
    "api_node": false,
    "price_badge": null,
    "search_aliases": null,
    "essentials_category": null,
    "has_intermediate_output": false
  },
  "BasicScheduler": {
    "input": {
      "required": {
        "model": [
          "MODEL",
          {}
        ],
        "scheduler": [
          "COMBO",
          {
            "multiselect": false,
            "options": [
              "simple",
              "sgm_uniform",
              "karras",
              "exponential",
              "ddim_uniform",
              "beta",
              "normal",
              "linear_quadratic",
              "kl_optimal"
            ]
          }
        ],
        "steps": [
          "INT",
          {
            "default": 20,
            "min": 1,
            "max": 10000
          }
        ],
        "denoise": [
          "FLOAT",
          {
            "default": 1.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01
          }
        ]
      }
    },
    "input_order": {
      "required": [
        "model",
        "scheduler",
        "steps",
        "denoise"
      ]
    },
    "is_input_list": false,
    "output": [
      "SIGMAS"
    ],
    "output_is_list": [
      false
    ],
    "output_name": [
      "SIGMAS"
    ],
    "output_tooltips": [
      null
    ],
    "output_matchtypes": null,
    "name": "BasicScheduler",
    "display_name": null,
    "description": "",
    "python_module": "comfy_extras.nodes_custom_sampler",
    "category": "model/sampling/schedulers",
    "output_node": false,
    "deprecated": false,
    "experimental": false,
    "dev_only": false,
    "api_node": false,
    "price_badge": null,
    "search_aliases": null,
    "essentials_category": null,
    "has_intermediate_output": false
  },
  "Flux2Scheduler": {
    "input": {
      "required": {
        "steps": [
          "INT",
          {
            "default": 20,
            "min": 1,
            "max": 4096
          }
        ],
        "width": [
          "INT",
          {
            "default": 1024,
            "min": 16,
            "max": 16384,
            "step": 1
          }
        ],
        "height": [
          "INT",
          {
            "default": 1024,
            "min": 16,
            "max": 16384,
            "step": 1
          }
        ]
      }
    },
    "input_order": {
      "required": [
        "steps",
        "width",
        "height"
      ]
    },
    "is_input_list": false,
    "output": [
      "SIGMAS"
    ],
    "output_is_list": [
      false
    ],
    "output_name": [
      "SIGMAS"
    ],
    "output_tooltips": [
      null
    ],
    "output_matchtypes": null,
    "name": "Flux2Scheduler",
    "display_name": null,
    "description": "",
    "python_module": "comfy_extras.nodes_flux",
    "category": "model/sampling/schedulers",
    "output_node": false,
    "deprecated": false,
    "experimental": false,
    "dev_only": false,
    "api_node": false,
    "price_badge": null,
    "search_aliases": null,
    "essentials_category": null,
    "has_intermediate_output": false
  },
  "SamplerCustomAdvanced": {
    "input": {
      "required": {
        "noise": [
          "NOISE",
          {}
        ],
        "guider": [
          "GUIDER",
          {}
        ],
        "sampler": [
          "SAMPLER",
          {}
        ],
        "sigmas": [
          "SIGMAS",
          {}
        ],
        "latent_image": [
          "LATENT",
          {}
        ]
      }
    },
    "input_order": {
      "required": [
        "noise",
        "guider",
        "sampler",
        "sigmas",
        "latent_image"
      ]
    },
    "is_input_list": false,
    "output": [
      "LATENT",
      "LATENT"
    ],
    "output_is_list": [
      false,
      false
    ],
    "output_name": [
      "output",
      "denoised_output"
    ],
    "output_tooltips": [
      null,
      null
    ],
    "output_matchtypes": null,
    "name": "SamplerCustomAdvanced",
    "display_name": null,
    "description": "",
    "python_module": "comfy_extras.nodes_custom_sampler",
    "category": "model/sampling/custom",
    "output_node": false,
    "deprecated": false,
    "experimental": false,
    "dev_only": false,
    "api_node": false,
    "price_badge": null,
    "search_aliases": null,
    "essentials_category": null,
    "has_intermediate_output": false
  },
  "SamplerLCM": {
    "input": {
      "required": {
        "s_noise": [
          "FLOAT",
          {
            "tooltip": "Per-step noise multiplier at the first step (1.0 = match training).",
            "default": 1.0,
            "min": 0.0,
            "max": 64.0,
            "step": 0.01
          }
        ],
        "s_noise_end": [
          "FLOAT",
          {
            "tooltip": "Per-step noise multiplier at the last step. Set equal to s_noise for a constant schedule.",
            "default": 1.0,
            "min": 0.0,
            "max": 64.0,
            "step": 0.01
          }
        ],
        "noise_clip_std": [
          "FLOAT",
          {
            "tooltip": "Clamp per-step noise to +/- N*std. 0 disables.",
            "default": 0.0,
            "min": 0.0,
            "max": 10.0,
            "step": 0.01
          }
        ]
      }
    },
    "input_order": {
      "required": [
        "s_noise",
        "s_noise_end",
        "noise_clip_std"
      ]
    },
    "is_input_list": false,
    "output": [
      "SAMPLER"
    ],
    "output_is_list": [
      false
    ],
    "output_name": [
      "SAMPLER"
    ],
    "output_tooltips": [
      null
    ],
    "output_matchtypes": null,
    "name": "SamplerLCM",
    "display_name": null,
    "description": "LCM sampler with tunable per-step noise. s_noise is a multiplier on the model's training noise scale",
    "python_module": "comfy_extras.nodes_advanced_samplers",
    "category": "model/sampling/samplers",
    "output_node": false,
    "deprecated": false,
    "experimental": false,
    "dev_only": false,
    "api_node": false,
    "price_badge": null,
    "search_aliases": null,
    "essentials_category": null,
    "has_intermediate_output": false
  },
  "ReferenceLatent": {
    "input": {
      "required": {
        "conditioning": [
          "CONDITIONING",
          {}
        ]
      },
      "optional": {
        "latent": [
          "LATENT",
          {}
        ]
      }
    },
    "input_order": {
      "required": [
        "conditioning"
      ],
      "optional": [
        "latent"
      ]
    },
    "is_input_list": false,
    "output": [
      "CONDITIONING"
    ],
    "output_is_list": [
      false
    ],
    "output_name": [
      "CONDITIONING"
    ],
    "output_tooltips": [
      null
    ],
    "output_matchtypes": null,
    "name": "ReferenceLatent",
    "display_name": "Set Reference Latent",
    "description": "This node sets the guiding latent for an edit model. If the model supports it you can chain multiple to set multiple reference images.",
    "python_module": "comfy_extras.nodes_edit_model",
    "category": "model/conditioning",
    "output_node": false,
    "deprecated": false,
    "experimental": false,
    "dev_only": false,
    "api_node": false,
    "price_badge": null,
    "search_aliases": null,
    "essentials_category": null,
    "has_intermediate_output": false
  },
  "ImageScaleToTotalPixels": {
    "input": {
      "required": {
        "image": [
          "IMAGE",
          {}
        ],
        "upscale_method": [
          "COMBO",
          {
            "multiselect": false,
            "options": [
              "nearest-exact",
              "bilinear",
              "area",
              "bicubic",
              "lanczos"
            ]
          }
        ],
        "megapixels": [
          "FLOAT",
          {
            "default": 1.0,
            "min": 0.01,
            "max": 16.0,
            "step": 0.01
          }
        ],
        "resolution_steps": [
          "INT",
          {
            "advanced": true,
            "default": 1,
            "min": 1,
            "max": 256
          }
        ]
      }
    },
    "input_order": {
      "required": [
        "image",
        "upscale_method",
        "megapixels",
        "resolution_steps"
      ]
    },
    "is_input_list": false,
    "output": [
      "IMAGE"
    ],
    "output_is_list": [
      false
    ],
    "output_name": [
      "IMAGE"
    ],
    "output_tooltips": [
      null
    ],
    "output_matchtypes": null,
    "name": "ImageScaleToTotalPixels",
    "display_name": "Scale Image to Total Pixels",
    "description": "",
    "python_module": "comfy_extras.nodes_post_processing",
    "category": "image/upscaling",
    "output_node": false,
    "deprecated": false,
    "experimental": false,
    "dev_only": false,
    "api_node": false,
    "price_badge": null,
    "search_aliases": null,
    "essentials_category": null,
    "has_intermediate_output": false
  },
  "GetImageSize": {
    "input": {
      "required": {
        "image": [
          "IMAGE",
          {}
        ]
      },
      "hidden": {
        "unique_id": [
          "UNIQUE_ID"
        ]
      }
    },
    "input_order": {
      "required": [
        "image"
      ],
      "hidden": [
        "unique_id"
      ]
    },
    "is_input_list": false,
    "output": [
      "INT",
      "INT",
      "INT"
    ],
    "output_is_list": [
      false,
      false,
      false
    ],
    "output_name": [
      "width",
      "height",
      "batch_size"
    ],
    "output_tooltips": [
      null,
      null,
      null
    ],
    "output_matchtypes": null,
    "name": "GetImageSize",
    "display_name": "Get Image Size",
    "description": "Returns the width, height, and batch size of the image.",
    "python_module": "comfy_extras.nodes_images",
    "category": "image",
    "output_node": false,
    "deprecated": false,
    "experimental": false,
    "dev_only": false,
    "api_node": false,
    "price_badge": null,
    "search_aliases": [
      "dimensions",
      "resolution",
      "image info"
    ],
    "essentials_category": null,
    "has_intermediate_output": false
  },
  "TextEncodeMingImageEdit": {
    "input": {
      "required": {
        "clip": [
          "CLIP",
          {}
        ],
        "prompt": [
          "STRING",
          {
            "multiline": true,
            "dynamicPrompts": true
          }
        ],
        "images": [
          "COMFY_AUTOGROW_V3",
          {
            "tooltip": "Optional reference images, seen by the text encoder and appended to the latent sequence as clean frames. Later images are resized to the first one, and the sampled latent should match its size.",
            "template": {
              "input": {
                "required": {
                  "image": [
                    "IMAGE",
                    {}
                  ]
                }
              },
              "names": [
                "image_1",
                "image_2",
                "image_3",
                "image_4",
                "image_5",
                "image_6",
                "image_7",
                "image_8"
              ],
              "min": 0
            }
          }
        ]
      },
      "optional": {
        "vae": [
          "VAE",
          {
            "tooltip": "Without a VAE the images only condition the text encoder through the vision tower."
          }
        ]
      }
    },
    "input_order": {
      "required": [
        "clip",
        "prompt",
        "images"
      ],
      "optional": [
        "vae"
      ]
    },
    "is_input_list": false,
    "output": [
      "CONDITIONING"
    ],
    "output_is_list": [
      false
    ],
    "output_name": [
      "CONDITIONING"
    ],
    "output_tooltips": [
      null
    ],
    "output_matchtypes": null,
    "name": "TextEncodeMingImageEdit",
    "display_name": "Text Encode Ming Image Edit",
    "description": "",
    "python_module": "comfy_extras.nodes_ming",
    "category": "model/conditioning/ming image",
    "output_node": false,
    "deprecated": false,
    "experimental": false,
    "dev_only": false,
    "api_node": false,
    "price_badge": null,
    "search_aliases": null,
    "essentials_category": null,
    "has_intermediate_output": false
  },
  "CLIPTextEncode": {
    "input": {
      "required": {
        "text": [
          "STRING",
          {
            "multiline": true,
            "dynamicPrompts": true,
            "tooltip": "The text to be encoded."
          }
        ],
        "clip": [
          "CLIP",
          {
            "tooltip": "The CLIP model used for encoding the text."
          }
        ]
      }
    },
    "input_order": {
      "required": [
        "text",
        "clip"
      ]
    },
    "is_input_list": false,
    "output": [
      "CONDITIONING"
    ],
    "output_is_list": [
      false
    ],
    "output_name": [
      "CONDITIONING"
    ],
    "name": "CLIPTextEncode",
    "display_name": "CLIP Text Encode (Prompt)",
    "description": "Encodes a text prompt using a CLIP model into an embedding that can be used to guide the diffusion model towards generating specific images.",
    "python_module": "nodes",
    "category": "model/conditioning",
    "output_node": false,
    "has_intermediate_output": false,
    "output_tooltips": [
      "A conditioning containing the embedded text used to guide the diffusion model."
    ],
    "search_aliases": [
      "text",
      "prompt",
      "text prompt",
      "positive prompt",
      "negative prompt",
      "encode text",
      "text encoder",
      "encode prompt"
    ]
  },
  "KSampler": {
    "input": {
      "required": {
        "model": [
          "MODEL",
          {
            "tooltip": "The model used for denoising the input latent."
          }
        ],
        "seed": [
          "INT",
          {
            "default": 0,
            "min": 0,
            "max": 18446744073709551615,
            "control_after_generate": true,
            "tooltip": "The random seed used for creating the noise."
          }
        ],
        "steps": [
          "INT",
          {
            "default": 20,
            "min": 1,
            "max": 10000,
            "tooltip": "The number of steps used in the denoising process."
          }
        ],
        "cfg": [
          "FLOAT",
          {
            "default": 8.0,
            "min": 0.0,
            "max": 100.0,
            "step": 0.1,
            "round": 0.01,
            "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."
          }
        ],
        "sampler_name": [
          [
            "euler",
            "euler_cfg_pp",
            "euler_ancestral",
            "euler_ancestral_cfg_pp",
            "heun",
            "heunpp2",
            "exp_heun_2_x0",
            "exp_heun_2_x0_sde",
            "dpm_2",
            "dpm_2_ancestral",
            "lms",
            "dpm_fast",
            "dpm_adaptive",
            "dpmpp_2s_ancestral",
            "dpmpp_2s_ancestral_cfg_pp",
            "dpmpp_sde",
            "dpmpp_sde_gpu",
            "dpmpp_2m",
            "dpmpp_2m_cfg_pp",
            "dpmpp_2m_sde",
            "dpmpp_2m_sde_gpu",
            "dpmpp_2m_sde_heun",
            "dpmpp_2m_sde_heun_gpu",
            "dpmpp_3m_sde",
            "dpmpp_3m_sde_gpu",
            "ddpm",
            "lcm",
            "ipndm",
            "ipndm_v",
            "deis",
            "cfgpp_ud10_ab",
            "res_multistep",
            "res_multistep_cfg_pp",
            "res_multistep_ancestral",
            "res_multistep_ancestral_cfg_pp",
            "gradient_estimation",
            "gradient_estimation_cfg_pp",
            "er_sde",
            "seeds_2",
            "seeds_3",
            "sa_solver",
            "sa_solver_pece",
            "ddim",
            "uni_pc",
            "uni_pc_bh2"
          ],
          {
            "tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."
          }
        ],
        "scheduler": [
          [
            "simple",
            "sgm_uniform",
            "karras",
            "exponential",
            "ddim_uniform",
            "beta",
            "normal",
            "linear_quadratic",
            "kl_optimal"
          ],
          {
            "tooltip": "The scheduler controls how noise is gradually removed to form the image."
          }
        ],
        "positive": [
          "CONDITIONING",
          {
            "tooltip": "The conditioning describing the attributes you want to include in the image."
          }
        ],
        "negative": [
          "CONDITIONING",
          {
            "tooltip": "The conditioning describing the attributes you want to exclude from the image."
          }
        ],
        "latent_image": [
          "LATENT",
          {
            "tooltip": "The latent image to denoise."
          }
        ],
        "denoise": [
          "FLOAT",
          {
            "default": 1.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."
          }
        ]
      }
    },
    "input_order": {
      "required": [
        "model",
        "seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "positive",
        "negative",
        "latent_image",
        "denoise"
      ]
    },
    "is_input_list": false,
    "output": [
      "LATENT"
    ],
    "output_is_list": [
      false
    ],
    "output_name": [
      "LATENT"
    ],
    "name": "KSampler",
    "display_name": "KSampler",
    "description": "Uses the provided model, positive and negative conditioning to denoise the latent image.",
    "python_module": "nodes",
    "category": "model/sampling",
    "output_node": false,
    "has_intermediate_output": false,
    "output_tooltips": [
      "The denoised latent."
    ],
    "search_aliases": [
      "sampler",
      "sample",
      "generate",
      "denoise",
      "diffuse",
      "txt2img",
      "img2img"
    ]
  },
  "EmptyLatentImage": {
    "input": {
      "required": {
        "width": [
          "INT",
          {
            "default": 1024,
            "min": 16,
            "max": 16384,
            "step": 8,
            "tooltip": "The width of the latent images in pixels."
          }
        ],
        "height": [
          "INT",
          {
            "default": 1024,
            "min": 16,
            "max": 16384,
            "step": 8,
            "tooltip": "The height of the latent images in pixels."
          }
        ],
        "batch_size": [
          "INT",
          {
            "default": 1,
            "min": 1,
            "max": 4096,
            "tooltip": "The number of latent images in the batch."
          }
        ]
      }
    },
    "input_order": {
      "required": [
        "width",
        "height",
        "batch_size"
      ]
    },
    "is_input_list": false,
    "output": [
      "LATENT"
    ],
    "output_is_list": [
      false
    ],
    "output_name": [
      "LATENT"
    ],
    "name": "EmptyLatentImage",
    "display_name": "Empty Latent Image",
    "description": "Create a new batch of empty latent images to be denoised via sampling.",
    "python_module": "nodes",
    "category": "model/latent",
    "output_node": false,
    "has_intermediate_output": false,
    "output_tooltips": [
      "The empty latent image batch."
    ],
    "search_aliases": [
      "empty",
      "empty latent",
      "new latent",
      "create latent",
      "blank latent",
      "blank"
    ]
  },
  "VAEEncode": {
    "input": {
      "required": {
        "pixels": [
          "IMAGE"
        ],
        "vae": [
          "VAE"
        ]
      }
    },
    "input_order": {
      "required": [
        "pixels",
        "vae"
      ]
    },
    "is_input_list": false,
    "output": [
      "LATENT"
    ],
    "output_is_list": [
      false
    ],
    "output_name": [
      "LATENT"
    ],
    "name": "VAEEncode",
    "display_name": "VAE Encode",
    "description": "",
    "python_module": "nodes",
    "category": "model/latent",
    "output_node": false,
    "has_intermediate_output": false,
    "search_aliases": [
      "encode",
      "encode image",
      "image to latent"
    ]
  },
  "VAEDecode": {
    "input": {
      "required": {
        "samples": [
          "LATENT",
          {
            "tooltip": "The latent to be decoded."
          }
        ],
        "vae": [
          "VAE",
          {
            "tooltip": "The VAE model used for decoding the latent."
          }
        ]
      }
    },
    "input_order": {
      "required": [
        "samples",
        "vae"
      ]
    },
    "is_input_list": false,
    "output": [
      "IMAGE"
    ],
    "output_is_list": [
      false
    ],
    "output_name": [
      "IMAGE"
    ],
    "name": "VAEDecode",
    "display_name": "VAE Decode",
    "description": "Decodes latent images back into pixel space images.",
    "python_module": "nodes",
    "category": "model/latent",
    "output_node": false,
    "has_intermediate_output": false,
    "output_tooltips": [
      "The decoded image."
    ],
    "search_aliases": [
      "decode",
      "decode latent",
      "latent to image",
      "render latent"
    ]
  },
  "SaveImage": {
    "input": {
      "required": {
        "images": [
          "IMAGE",
          {
            "tooltip": "The images to save."
          }
        ],
        "filename_prefix": [
          "STRING",
          {
            "default": "ComfyUI",
            "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."
          }
        ]
      },
      "hidden": {
        "prompt": "PROMPT",
        "extra_pnginfo": "EXTRA_PNGINFO"
      }
    },
    "input_order": {
      "required": [
        "images",
        "filename_prefix"
      ],
      "hidden": [
        "prompt",
        "extra_pnginfo"
      ]
    },
    "is_input_list": false,
    "output": [
      "IMAGE"
    ],
    "output_is_list": [
      false
    ],
    "output_name": [
      "images"
    ],
    "name": "SaveImage",
    "display_name": "Save Image",
    "description": "Saves the input images to your ComfyUI output directory.",
    "python_module": "nodes",
    "category": "image",
    "output_node": true,
    "has_intermediate_output": false,
    "search_aliases": [
      "save",
      "save image",
      "export image",
      "output image",
      "write image",
      "download"
    ],
    "essentials_category": "Basics"
  }
} as unknown as ObjectInfo;
