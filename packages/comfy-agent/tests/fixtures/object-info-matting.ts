import type { ObjectInfo } from "../../src/types.js";

// Node definitions copied verbatim from a live ComfyUI 0.43.0 server's
// `/object_info` (commit 79cf64c, Apple silicon, with Comfy-Org/BiRefNet
// `birefnet.safetensors` in models/background_removal). These are core nodes:
// background removal needs no custom node.

export const MATTING_NODES: ObjectInfo = {
  "LoadBackgroundRemovalModel": {
    "input": {
      "required": {
        "bg_removal_name": [
          "COMBO",
          {
            "tooltip": "The model used to remove backgrounds from images",
            "multiselect": false,
            "options": [
              "birefnet.safetensors"
            ]
          }
        ]
      }
    },
    "input_order": {
      "required": [
        "bg_removal_name"
      ]
    },
    "output": [
      "BACKGROUND_REMOVAL"
    ],
    "output_is_list": [
      false
    ],
    "output_name": [
      "bg_model"
    ],
    "name": "LoadBackgroundRemovalModel",
    "display_name": "Load Background Removal Model",
    "category": "model/loaders",
    "output_node": false,
    "api_node": false
  },
  "RemoveBackground": {
    "input": {
      "required": {
        "bg_removal_model": [
          "BACKGROUND_REMOVAL",
          {
            "tooltip": "Background removal model used to generate the mask"
          }
        ],
        "image": [
          "IMAGE",
          {
            "tooltip": "Input image to remove the background from"
          }
        ]
      }
    },
    "input_order": {
      "required": [
        "bg_removal_model",
        "image"
      ]
    },
    "output": [
      "MASK"
    ],
    "output_is_list": [
      false
    ],
    "output_name": [
      "mask"
    ],
    "name": "RemoveBackground",
    "display_name": "Remove Background",
    "category": "image/background removal",
    "output_node": false,
    "api_node": false
  },
  "InvertMask": {
    "input": {
      "required": {
        "mask": [
          "MASK",
          {}
        ]
      }
    },
    "input_order": {
      "required": [
        "mask"
      ]
    },
    "output": [
      "MASK"
    ],
    "output_is_list": [
      false
    ],
    "output_name": [
      "MASK"
    ],
    "name": "InvertMask",
    "display_name": "Invert Mask",
    "category": "image/mask",
    "output_node": false,
    "api_node": false
  },
  "JoinImageWithAlpha": {
    "input": {
      "required": {
        "image": [
          "IMAGE",
          {}
        ],
        "alpha": [
          "MASK",
          {}
        ]
      }
    },
    "input_order": {
      "required": [
        "image",
        "alpha"
      ]
    },
    "output": [
      "IMAGE"
    ],
    "output_is_list": [
      false
    ],
    "output_name": [
      "IMAGE"
    ],
    "name": "JoinImageWithAlpha",
    "display_name": "Join Image with Alpha",
    "category": "image/compositing",
    "output_node": false,
    "api_node": false
  }
};
