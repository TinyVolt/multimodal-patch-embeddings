from torch.utils.data import Dataset
from glob import glob
from PIL import Image
from torchvision import transforms
from typing import List, Dict
import torch
import torch.nn as nn

from utils import open_file_in_rgb_format, find_all_images_in_directory, keep_top_n_files_containing_pattern

mu = torch.FloatTensor([0.48145466, 0.4578275, 0.40821073])
sigma = torch.FloatTensor([0.26862954, 0.26130258, 0.27577711])

transform_train = transforms.Compose([
    transforms.Resize((300,300)),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.RandomApply([transforms.GaussianBlur(5)], p=0.05),
    transforms.RandomPosterize(3, p=0.005),
    transforms.RandomAutocontrast(p=0.001),
    transforms.RandomCrop(size=(224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mu, std=sigma)
])
transform_val = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mu, std=sigma)
])

pattern_to_max_num_files_dict: Dict[str, int] = {}


class ClipDataset(Dataset):
    def __init__(self, filenames:List[str], transform: nn.Module, pattern_to_max_num_files_dict: Dict[str, int] = pattern_to_max_num_files_dict):
        self.filenames = filenames
        self.transform = transform
        # Some folders contain many images of the same type. To prevent overfitting on these types of images, we limit their number.
        self.pattern_to_max_num_files_dict = pattern_to_max_num_files_dict
        for (pattern, max_num_files) in pattern_to_max_num_files_dict.items():
            self.filenames = keep_top_n_files_containing_pattern(self.filenames, pattern, max_num_files)

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        image = open_file_in_rgb_format(self.filenames[index])
        return self.transform(image)
