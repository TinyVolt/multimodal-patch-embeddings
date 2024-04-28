# multimodal-patch-embeddings

![](https://github.com/TinyVolt/multimodal-patch-embeddings/blob/0d5783fe809cfef407086cae0f8d7a80748bc950/assets/one.gif)
![](https://github.com/TinyVolt/multimodal-patch-embeddings/blob/0d5783fe809cfef407086cae0f8d7a80748bc950/assets/four.gif)

## Directory structure
```
.
├── LICENSE
├── README.md
├── assets
│   ├── attention_comparison_no_cls
│   ├── patch_activations
│   └── search_comparison_no_cls
├── checkpoints
│   ├── checkpoint_epoch24_vit_extended_dim_2024-04-11_19-18-30.pt
│   └── checkpoint_epoch31_vit_extended_dim_same_norm_attn_mask_2024-04-27_20-29-33.pt
├── poetry.lock
├── pyproject.toml
└── src
    ├── _types.py
    ├── data.py
    ├── loss.py
    ├── main.py
    ├── model.py
    ├── notebooks
    │   ├── mm_patch_embed.ipynb
    │   └── vit_no_cls.ipynb
    └── utils.py
```

## Downloading the checkpoints
- Download the checkpoints from [here](https://huggingface.co/vinsis/multimodal-patch-embeddings) and put them in the `checkpoints` folder.

## Setting up
```
poetry install
```

## Results

![](https://github.com/TinyVolt/multimodal-patch-embeddings/blob/6f6bf04aa2a73c8c2bc585c9f11d4158fbe3602b/assets/patch_activations/1_combined.jpg)
![](https://github.com/TinyVolt/multimodal-patch-embeddings/blob/6f6bf04aa2a73c8c2bc585c9f11d4158fbe3602b/assets/patch_activations/3_combined.jpg)
![](https://github.com/TinyVolt/multimodal-patch-embeddings/blob/6f6bf04aa2a73c8c2bc585c9f11d4158fbe3602b/assets/patch_activations/4_combined.jpg)
![](https://github.com/TinyVolt/multimodal-patch-embeddings/blob/6f6bf04aa2a73c8c2bc585c9f11d4158fbe3602b/assets/patch_activations/6_combined.jpg)
![](https://github.com/TinyVolt/multimodal-patch-embeddings/blob/6f6bf04aa2a73c8c2bc585c9f11d4158fbe3602b/assets/patch_activations/7_combined.jpg)