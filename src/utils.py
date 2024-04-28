import torch
import numpy as np
from PIL import Image
import os
from _types import PretrainedViTNames, vit_name_to_arg_dict, ViTInput
from model import VisionTransformer
from typing import Dict, Tuple, List, Optional
from datetime import datetime
import argparse
import numpy as np
from glob import glob

PWD = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PWD, '..', 'checkpoints')
ASSETS_DIR = os.path.join(PWD, '..', 'assets')

WHITE_IMAGE = Image.new(mode='RGB', size=(224,224), color=255)

def visualize_attention_patches(attention_probs: torch.Tensor, image: Optional[Image.Image] = None) -> Image.Image:
    '''
    Visualize attention patches by overlaying them on the original image.
    Inputs:
    - `attention_probs`: tensor of shape `(grid_size, grid_size)`
    - `image`: PIL Image object of shape (dim, dim, 3)
    Outputs:
    - `overlay`: PIL Image object
    '''
    attention_probs = attention_probs / attention_probs.max()
    mode = 'L' if image is None else 'RGB'
    image = image or WHITE_IMAGE
    image_array = np.array(image) # (dim, dim, C=3)
    grid_size = attention_probs.shape[0]
    image_dim = image_array.shape[0]
    patch_size = image_dim // grid_size
    pixelwise_probs = torch.kron(attention_probs, torch.ones(patch_size, patch_size)).unsqueeze(-1).numpy() # (dim, dim, 1)
    overlay = (image_array * pixelwise_probs).astype(np.uint8) # (dim, dim, C=3)
    return Image.fromarray(overlay).convert(mode).convert('RGB')

def load_pretrained_vit(
    vit_name:PretrainedViTNames, 
    checkpoint_dir = os.path.expanduser('~/.cache/clip/vit_state_dicts'),
    vit_name_to_arg_dict: Dict[PretrainedViTNames, ViTInput] = vit_name_to_arg_dict
):
    checkpoint_file = os.path.join(checkpoint_dir, f'{vit_name}.pt')
    assert os.path.exists(checkpoint_file), f'Checkpoint file {checkpoint_file} does not exist'
    vit_args = vit_name_to_arg_dict[vit_name]
    vit = VisionTransformer(**vit_args.model_dump())
    vit.load_state_dict(torch.load(checkpoint_file, map_location='cpu'))
    return vit.eval()

def _get_date_time() -> str:
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def save_checkpoint(dict_to_save: Dict, epoch: int, checkpoint_dir: str = CHECKPOINT_DIR, suffix=''):
    os.makedirs(checkpoint_dir, exist_ok=True)
    if len(suffix) > 0:
        suffix = suffix + '_'
    checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}_{suffix}{_get_date_time()}.pt')
    torch.save(dict_to_save, checkpoint_file)
    print(f'Checkpoint for epoch {epoch} saved to {checkpoint_file}')

def get_hyperparam_args():
    parser = argparse.ArgumentParser(description='Hyperparams to train a Vision Transformer')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--loss_weight', type=float, default=20, help='Weight for the attention loss')
    parser.add_argument('--loss_norm', default=20, type=float, help='norm to be used in normalized l1 loss')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume training from')
    parser.add_argument('--train_folder', type=str, default='xxx' , help='Path to folder containing training images')
    parser.add_argument('--val_folder', type=str, default='xxx', help='Path to folder containing val images')
    parser.add_argument('--print_every', type=int, default=10, help='Print loss every n iterations')
    parser.add_argument('--no_resume_opt_sch', action='store_true', help='Do not load the state dicts of optimizer and scheduler from the checkpoint')
    return parser.parse_args()

def _find_all_files_in_directory(directory: str, extensions: Tuple[str, ...]) -> list:
    ext_regex = '[.' + '|.'.join(extensions) + ']'
    result = glob(os.path.join(directory, '**', f'*{ext_regex}'), recursive=True)
    return [f for f in result if f.endswith(extensions)]

def find_all_images_in_directory(directory: str, extensions: Tuple[str, ...] = ('jpg', 'jpeg', 'png')):
    extensions += tuple(ext.upper() for ext in extensions)
    return _find_all_files_in_directory(directory, extensions)

def open_file_in_rgb_format(filename: str):
    '''
    TODO: Replace this hacky way of dealing with corrupt images with something more robust
    '''
    default_filename = os.path.join(ASSETS_DIR, 'cat.jpg')
    try:
        image = Image.open(filename).convert('RGB')
    except:
        print(f'something went wrong with {filename}, using default fn')
        image = Image.open(default_filename).convert('RGB')
    return image

def keep_top_n_files_containing_pattern(filenames: List[str], pattern: str, n: int) -> List[str]:
    files_without_pattern, files_with_pattern = [], []
    for fn in filenames:
        if pattern in fn:
            files_with_pattern.append(fn)
        else:
            files_without_pattern.append(fn)
    return files_with_pattern[:n] + files_without_pattern

def randint(low: int, high: int) -> int:
    return np.random.randint(low, high)

def random_color() -> Tuple[int, int, int]:
    return (randint(0,256), randint(0,256), randint(0,256))

def split_files_into_train_and_val(files:List[str], p:float=0.9, max_val_size:int=10000) -> Tuple[List[str], List[str]]:
    val_size = min(max_val_size, int((1-p) * len(files)))
    train_size = len(files) - val_size
    indices = np.random.permutation(len(files))
    split_train = np.array(files)[indices[:train_size]]
    split_val = np.array(files)[indices[train_size:]]
    return split_train.tolist(), split_val.tolist()

def stitch_images(images:List[Image.Image], horizontal:bool=True) -> Image.Image:
    images = [ np.array(image.convert('RGB')) for image in images ]
    axis = 1 if horizontal else 0
    return Image.fromarray(np.concatenate(images, axis=axis))

def get_top_n_results(text_vector:torch.Tensor, image_embeds:torch.Tensor, n:int=10) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    - `text_vector` has shape (1, dim)
    - `image_embeds` have shape (N, dim)
    '''
    scores = (text_vector.float() @ image_embeds.t()).squeeze()
    top_n_scores, top_n_indices = scores.topk(n)
    return top_n_scores, top_n_indices