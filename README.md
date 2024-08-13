# multimodal-patch-embeddings

![](https://github.com/TinyVolt/multimodal-patch-embeddings/blob/0d5783fe809cfef407086cae0f8d7a80748bc950/assets/one.gif)
![](https://github.com/TinyVolt/multimodal-patch-embeddings/blob/0d5783fe809cfef407086cae0f8d7a80748bc950/assets/four.gif)
![](https://github.com/TinyVolt/multimodal-patch-embeddings/blob/cf93109e17ea0fa4f663f8786edddebc10e7ac17/assets/six.gif)
![](https://github.com/TinyVolt/multimodal-patch-embeddings/blob/cf93109e17ea0fa4f663f8786edddebc10e7ac17/assets/three.gif)

## Details
This repo contains the code for training and inference of distilled, and smaller, CLIP ViT model. The distilled model has 21.3 million parameters. The vision transformer uses a novel architecture which is much simpler. It does not have CLS embedding, neither does it have a projection layer at the end. Check out the `src/mode.py/VisionTransformerExtraHead` class to see the implementation. Check out the article here: https://www.tinyvolt.com/research/multimodal-patch-embeddings

## Multimodal patch embeddings
What makes this model so special is that the embedding of each of the image patches is in the same embedding space as the final embedding. In fact, the final embedding is just a convex sum of the patch embeddings. This allows one to compare the text embedding with each of the 64 image patch embeddings.

## Model output(s)
The ViT model maps an image to an embedding. By default, the model outputs the embedding (of shape `B,512`) and the probability distribution over the image patches (of shape (`B,1,64`) where 64 is the number of patches). However if you want to get the embedding for each image patch, you just need to pass an extra parameter, `return_all_embeds`, during inference:

```python
import torch

# make sure you are in `src/` directory
from model import VisionTransformerExtraHead
from _types import vit_extended_same_norm_masked_28_args_16_heads_512_width as vit_multimodal_patch_args
vit = VisionTransformerExtraHead(**vit_multimodal_patch_args.model_dump())

x = torch.randn(1,3,224,224)
with torch.no_grad():
    y, attn = vit(x)
    print(y.shape, attn.shape)
    y, attn = vit(x, return_all_embeds=True)
    print(y.shape, attn.shape)
```

This will print the following:
```python
torch.Size([1, 512]) torch.Size([1, 1, 64])
torch.Size([1, 64, 512]) torch.Size([1, 1, 64])
```

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

The checkpoint `checkpoint_epoch24_vit_extended_dim_2024-04-11_19-18-30.pt` was not trained with the attention mask. It also does not enforce the patch embeddings to have the same norm before taking a convex sum. As a result, it does not need (and contain) the `scale` parameter defined in the `VisionTransformerExtraHead` class. To load this checkpoint, you can do something like so:

```python
import torch

# make sure you are in `src/` directory
from model import VisionTransformerExtraHead
from _types import vit_extended_28_args_16_heads_512_width as vit_no_cls
vit = VisionTransformerExtraHead(**vit_no_cls.model_dump())

# during inference, make sure to set `same_norm` to `False`
x = torch.randn(1,3,224,224)
y, attn = vit(x, same_norm=False, return_all_embeds=False)
```

Please note that this checkpoint does not have multimodal patch embeddings. 

## Setting up
```
poetry install
```

## Results
The below images show patch activations for different prompts.

![](https://github.com/TinyVolt/multimodal-patch-embeddings/blob/6f6bf04aa2a73c8c2bc585c9f11d4158fbe3602b/assets/patch_activations/1_combined.jpg)
![](https://github.com/TinyVolt/multimodal-patch-embeddings/blob/6f6bf04aa2a73c8c2bc585c9f11d4158fbe3602b/assets/patch_activations/3_combined.jpg)
![](https://github.com/TinyVolt/multimodal-patch-embeddings/blob/6f6bf04aa2a73c8c2bc585c9f11d4158fbe3602b/assets/patch_activations/4_combined.jpg)
![](https://github.com/TinyVolt/multimodal-patch-embeddings/blob/6f6bf04aa2a73c8c2bc585c9f11d4158fbe3602b/assets/patch_activations/6_combined.jpg)
![](https://github.com/TinyVolt/multimodal-patch-embeddings/blob/6f6bf04aa2a73c8c2bc585c9f11d4158fbe3602b/assets/patch_activations/7_combined.jpg)