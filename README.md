# multimodal-patch-embeddings

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