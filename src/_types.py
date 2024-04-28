import torch
from pydantic import BaseModel, PositiveInt
from enum import StrEnum
from typing import Dict, Optional


class ViTInput(BaseModel):
    input_resolution: PositiveInt
    patch_size: PositiveInt
    width: PositiveInt
    layers: PositiveInt
    heads: PositiveInt
    output_dim: PositiveInt
    attn_mask: Optional[torch.Tensor] = None

    class Config:
        arbitrary_types_allowed = True

###########################
## pretrained vit models ##
###########################

vit_b_16_args = ViTInput(
    input_resolution=224,
    patch_size=16,
    width=768,
    layers=12,
    heads=12,
    output_dim=512
)

vit_b_32_args = ViTInput(
    input_resolution=224,
    patch_size=32,
    width=768,
    layers=12,
    heads=12,
    output_dim=512
)

vit_l_14_args = ViTInput(
    input_resolution=224,
    patch_size=14,
    width=1024,
    layers=24,
    heads=16,
    output_dim=768
)

vit_l_14_336px_args = ViTInput(
    input_resolution=336,
    patch_size=14,
    width=1024,
    layers=24,
    heads=16,
    output_dim=768
)

####################
## student models ##
####################

# TODO: find a better way to write this
def get_attn_mask(dim:int = 64) -> torch.FloatTensor:
    '''
    Returns a 2D attention mask for a given dimension.
    The mask is allows each element to attend only to its immediate neighbours.
    '''
    width = int(dim**0.5)
    assert width**2 == dim, f'dim should be a perfect square but {dim} was provided'
    output = torch.ones(dim, dim) * -torch.inf
    for i in range(dim):
        if i%width == 0:
            indices = [i,i+1,i+1+width,i+1-width,i-width,i+width]
        elif i%width == width-1:
            indices = [i-1,i,i-1+width,i-1-width,i-width,i+width]
        else:
            indices = [i-1,i,i+1,i-width,i+width,i-1-width,i-1+width,i+1-width,i+1+width]
        indices = torch.LongTensor([index for index in indices if 0 <= index < dim])
        output[i, indices] = 0
    return output

vit_extended_28_args_16_heads_512_width = ViTInput(
    input_resolution=224,
    patch_size=28,
    width=512,
    layers=6,
    heads=16,
    output_dim=512
)

vit_extended_same_norm_masked_28_args_16_heads_512_width = ViTInput(
    input_resolution=224,
    patch_size=28,
    width=512,
    layers=6,
    heads=16,
    output_dim=512,
    attn_mask=get_attn_mask()
)

####################################
## names of pretrained vit models ##
####################################

PretrainedViTNames = StrEnum('PretrainedViTNames', ['vit_b_16', 'vit_b_32', 'vit_l_14', 'vit_l_14_336px'])

vit_name_to_arg_dict: Dict[PretrainedViTNames, ViTInput] = {
    PretrainedViTNames.vit_b_16: vit_b_16_args,
    PretrainedViTNames.vit_b_32: vit_b_32_args,
    PretrainedViTNames.vit_l_14: vit_l_14_args,
    PretrainedViTNames.vit_l_14_336px: vit_l_14_336px_args
}
