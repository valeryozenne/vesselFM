"""Adapted from https://github.com/mobaidoctor/med-ddpm/blob/main/sample.py"""

import re
import os
import math
import random
import argparse
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from PIL import Image, ImageDraw, ImageFont

from  vesselfm.d_flow.diffusion import FlowMatching
from  vesselfm.d_flow.diffusion_unet import create_model

warnings.filterwarnings("ignore")


def read_nifti(path):
    sitk_img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(sitk_img)


@torch.inference_mode()
def sample(args, device, rank, world_size): # [MODIFICATION] Ajout rank et world_size
    model = create_model(
        image_size=128, num_channels=64, num_res_blocks=1, in_channels=2, out_channels=1, 
        class_cond=args.class_cond, num_classes=args.num_classes, use_cfg=args.use_cfg
    ).to(device=device)

    diffusion = FlowMatching(
        model, use_lognorm_is=True, lognorm_mu=0, lognorm_sigma=1, num_steps=100,
        use_cfg=args.use_cfg, cfg_weight=args.cfg_weight, cfg_bg=args.cfg_bg, cfg_rand=args.cfg_rand,
        interpolate=args.interpolate, num_classes=args.num_classes
    ).to(device=device)
    
    ckpt = torch.load(args.ckpt_path, map_location="cpu") # [MODIFICATION] Load sur CPU d'abord pour éviter la surcharge VRAM
    diffusion.load_state_dict(ckpt[args.state_dict])
    model.to(device) # Transfert final sur le bon GPU

    # [MODIFICATION] Tri obligatoire avant shuffle pour que tous les GPUs aient la même base
    masks = sorted(list(Path(args.mask_folder).iterdir()))
    random.shuffle(masks)   # each seed should use different masks

    # [MODIFICATION] Division des masques : chaque GPU prend une fraction différente
    masks = masks[rank::world_size]
    
    # [MODIFICATION] Ajustement du nombre d'échantillons par GPU
    local_num_samples = math.ceil(args.num_samples / world_size)

    gen_samples = 0
    for mask in masks:
        m = torch.tensor(np.load(mask / "mask.npy"))[None][None].to(device=device).float()

        for class_id in range(args.num_classes):
            if args.production: # randomly select a class
                class_id = torch.randint(0, args.num_classes, (1,)).item()

            class_tensor = torch.tensor(class_id)[None].to(device=device)
            img = diffusion.sample(
                condition_tensors=m.repeat(args.batchsize, 1, 1, 1, 1),
                y=class_tensor.repeat(args.batchsize)
            )

            # [MODIFICATION] Calcul d'un ID unique global pour éviter que les GPUs n'écrivent dans le même dossier
            global_sample_id = args.start_id + (gen_samples * world_size) + rank

            if rank == 0: # Optionnel: n'afficher les logs que sur le GPU principal
                print(f'[Rank {rank}] Finished sample {global_sample_id} of class {class_id}.')
            
            path_to_sample = Path(args.out_folder) / str(global_sample_id)
            path_to_sample.mkdir(exist_ok=True, parents=True)

            if not args.no_npy:
                np.save(path_to_sample / f'mask_{class_id}.npy', m.cpu().squeeze().numpy().astype(np.bool_))
                np.save(path_to_sample / f'img_{class_id}.npy', img.cpu().squeeze().numpy().astype(np.float16))
            
            if args.nifti:
                nifti_img = nib.Nifti1Image(img.float().cpu().numpy().squeeze(), affine=np.eye(4))
                nib.save(nifti_img, str(path_to_sample / f'img_{class_id}.nii.gz'))
                nifti_mask = nib.Nifti1Image(m.int().cpu().numpy().squeeze(), affine=np.eye(4))
                nib.save(nifti_mask, str(path_to_sample / f'mask_{class_id}.nii.gz'))
            
            if args.production: # just one image per mask
                break

        if args.overview and args.nifti and not args.production:   # generate overview images
            generate_overview(
                sorted(
                    [p for p in path_to_sample.iterdir() if 'img' in str(p)],
                    key=lambda p: int(re.search(r'\d+', p.name).group())
                )
            )

        gen_samples += 1
        if gen_samples >= local_num_samples:
            break

def generate_overview(nifti_paths, font_path=None):
    images = [read_nifti(nifti) * 255 for nifti in nifti_paths]
    font = ImageFont.load_default() if font_path is None else ImageFont.truetype(font_path, 12)
    
    img_size = 128
    cols = 5
    rows = (len(images) + cols - 1) // cols
    padding = 15  # space for text

    for slice_ in [0, 31, 63, 95, 127]:
        combined_img = Image.new('L', (cols * img_size, rows * (img_size + padding)), 255)
        draw = ImageDraw.Draw(combined_img)

        for i, img in enumerate(images):
            x = (i % cols) * img_size
            y = (i // cols) * (img_size + padding)
            combined_img.paste(Image.fromarray(img[slice_]), (x, y))
            
            name = nifti_paths[i].name + f' slice {slice_}'

            text_bbox = draw.textbbox((0, 0), name, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = x + (img_size - text_width) // 2

            draw.text((text_x, y + img_size), name, font=font, fill=0)
    
        combined_img.save(nifti_paths[0].parent / f'overview_{slice_}.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--mask_folder', type=str, required=True)
    parser.add_argument('--out_folder', type=str, required=True)

    parser.add_argument('--no_npy', action='store_true')
    parser.add_argument('--nifti', action='store_true')
    parser.add_argument('--overview', action='store_true')
    parser.add_argument('--production', action='store_true')

    parser.add_argument('--state_dict', type=str, default='ema')
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--start_id', type=int, default=0)
    parser.add_argument('--timesteps', type=int, default=250)
    parser.add_argument('--seed', type=int, default=-1)

    parser.add_argument('--class_cond', action='store_true')
    parser.add_argument('--num_classes', type=int, default=24)

    parser.add_argument('--use_cfg', action='store_true')
    parser.add_argument('--cfg_weight', type=float, default=5)
    parser.add_argument('--cfg_bg', action='store_true')
    parser.add_argument('--cfg_rand', action='store_true')
    parser.add_argument('--interpolate', action='store_true')

    parser.add_argument('--gpu_id', type=str, default='0')
    args = parser.parse_args()

    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        device = torch.device('cuda:0')
        rank = 0
        world_size = 1

    if args.seed == -1:
        seed = 42 
    else:
        seed = args.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    sample(args, device, rank, world_size)

    if dist.is_initialized():
        dist.destroy_process_group()