import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from model import VisionTransformerExtraHead
from _types import PretrainedViTNames, vit_name_to_arg_dict, vit_extended_same_norm_masked_28_args_16_heads_512_width as vit_config
from utils import load_pretrained_vit, get_hyperparam_args, save_checkpoint, find_all_images_in_directory, split_files_into_train_and_val
from data import ClipDataset, transform_train, transform_val
from loss import calculate_normalized_l1_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def train(epoch, teacher_model: nn.Module, student_model: nn.Module, train_loader, optimizer, device, hyperparam_args):
    print(f'Epoch: {epoch}, Training...')
    teacher_model.eval()
    student_model.train()
    total_loss = 0
    for batch_idx, x in enumerate(train_loader):
        x = x.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            y_target, _ = teacher_model(x)
        y_pred, _ = student_model(x)
        loss = calculate_normalized_l1_loss(y_pred, y_target, norm=hyperparam_args.loss_norm)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        iterations = batch_idx * hyperparam_args.batch_size + x.size(0)
        if batch_idx % hyperparam_args.print_every == 0:
            print(f'Epoch: {epoch}, batch index: {batch_idx}, loss in current batch: {loss.item()}, loss_y_so_far: {total_loss/iterations}, scale: {student_model.scale.item()}')
        
    return total_loss / len(train_loader.dataset), None

def validate(epoch, teacher_model: nn.Module, student_model: nn.Module, val_loader, device):
    print(f'Epoch: {epoch}, Validating...')
    teacher_model.eval()
    student_model.eval()
    total_loss = 0
    with torch.no_grad():
        for _, x in enumerate(val_loader):
            x = x.to(device)
            y_target, _ = teacher_model(x)
            y_pred, _ = student_model(x)
            loss = calculate_normalized_l1_loss(y_pred, y_target, norm=hyperparam_args.loss_norm)
            total_loss += loss.item() * x.size(0)
    
    print(f'Epoch: {epoch}, Validation loss: {total_loss / len(val_loader.dataset)}')
    return total_loss / len(val_loader.dataset), None
            
if __name__ == '__main__':
    hyperparam_args = get_hyperparam_args()
    teacher_vit_name = PretrainedViTNames.vit_b_32
    teacher_vit_args = vit_name_to_arg_dict[teacher_vit_name]
    # get the models
    teacher_model = load_pretrained_vit(teacher_vit_name).to(device)
    student_model = VisionTransformerExtraHead(**vit_config.model_dump()).to(device)
    # calculate grid sizes
    grid_size_teacher = teacher_vit_args.input_resolution // teacher_vit_args.patch_size
    grid_size_student = vit_config.input_resolution // vit_config.patch_size
    teacher_model.grid_size = grid_size_teacher
    student_model.grid_size = grid_size_student
    # set up optimizer and scheduler
    optimizer = AdamW(student_model.parameters(), lr=hyperparam_args.lr, weight_decay=hyperparam_args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5, threshold=1e-3, min_lr=1e-6)
    # set up data loader
    # TODO: this part always ends up being messy, find a way to keep it tidy
    train_filenames = find_all_images_in_directory(hyperparam_args.train_folder)
    val_filenames = find_all_images_in_directory(hyperparam_args.val_folder)

    additional_train_filenames, additional_val_filenames = [], []
    imagenet_sketch_train, imagenet_sketch_val = split_files_into_train_and_val( find_all_images_in_directory('xxx') )
    tg_stickers_train, tg_stickers_val = split_files_into_train_and_val( find_all_images_in_directory('xxx') )
    additional_train_filenames += imagenet_sketch_train + tg_stickers_train
    additional_val_filenames += imagenet_sketch_val + tg_stickers_val
    train_filenames += additional_train_filenames
    val_filenames += additional_val_filenames

    train_dataset = ClipDataset(train_filenames, transform_train)
    val_dataset = ClipDataset(val_filenames, transform_val)
    print(f'# training images: {len(train_dataset)}, # val images: {len(val_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=hyperparam_args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hyperparam_args.batch_size)
    
    losses_y_train, losses_attn_train = [], []

    start_epoch = 0

    if hyperparam_args.resume:
        print(f'Resuming training from checkpoint {hyperparam_args.resume}')
        checkpoint = torch.load(hyperparam_args.resume)
        student_model.load_state_dict(checkpoint['model_state_dict'])
        if not hyperparam_args.no_resume_opt_sch:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # for group in optimizer.param_groups:
            #     group['lr'] = 5e-5
            print('optimizer lr is', [g['lr'] for g in optimizer.param_groups])
            # scheduler.threshold = 1e-2
        else:
            print('`no_resume_opt_sch` is set to True, will not load the state dicts of optimizer and scheduler')
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, hyperparam_args.epochs):
        loss_y_train, _ = train(epoch, teacher_model, student_model, train_loader, optimizer, device, hyperparam_args)
        losses_y_train.append(loss_y_train)
        print(f'Finished training epoch: {epoch}, loss_y: {loss_y_train}')
        loss_y_val, _ = validate(epoch, teacher_model, student_model, val_loader, device)
        scheduler.step(loss_y_val)
        dict_to_save = {
            'epoch': epoch,
            'model_state_dict': student_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss_y_train': loss_y_train,
            'loss_y_val': loss_y_val,
            'teacher_model_args': teacher_vit_name
        }
        save_checkpoint(dict_to_save, epoch, suffix='vit_extended_dim_same_norm_attn_mask')
        print('Scheduler state dict:', scheduler.state_dict())
    