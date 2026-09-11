""" Script to perform inference with vesselFM."""

import logging
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
import hydra
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from monai.inferers import SlidingWindowInfererAdapt
from skimage.morphology import remove_small_objects
from skimage.exposure import equalize_hist

from vesselfm.seg.utils.data import generate_transforms
from vesselfm.seg.utils.evaluation import Evaluator, calculate_mean_metrics


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def load_model(cfg, device):
    try:
        logger.info(f"Loading model from {cfg.ckpt_path}.")
        ckpt = torch.load(Path(cfg.ckpt_path), map_location=device, weights_only=False)
    except Exception as e:
        logger.warning(f"Failed to load model from {cfg.ckpt_path} due to {e}. Attempting to load from Hugging Face Hub.")
        logger.info(f"Loading model from Hugging Face.")
        hf_hub_download(repo_id='bwittmann/vesselFM', filename='meta.yaml') # required to track downloads
        ckpt = torch.load(
            hf_hub_download(repo_id='bwittmann/vesselFM', filename='vesselFM_base.pt'),
            map_location=device, weights_only=False
        )

    model = hydra.utils.instantiate(cfg.model)
    
    # 1. On extrait le dictionnaire des poids s'il vient de PyTorch Lightning
    state_dict = ckpt.get("state_dict", ckpt)
    
    # 2. (Optionnel mais recommandé) PyTorch Lightning ajoute souvent un préfixe 
    # "model." ou "net." aux noms des couches. On le retire pour que ça corresponde au DynUNet.
    clean_state_dict = {}
    for k, v in state_dict.items():
        # Enlève le préfixe s'il existe
        new_key = k.replace("model.", "").replace("net.", "") 
        clean_state_dict[new_key] = v

    # 3. On charge les poids nettoyés
    model.load_state_dict(clean_state_dict)
    
    return model

def write_nifti(data, file_path, spacing=None):
    meta_data = {}
    meta_data['itk_spacing'] = spacing if spacing else [1, 1, 1]
    data_itk = sitk.GetImageFromArray(data)
    data_itk.SetSpacing(meta_data['itk_spacing'])
    sitk.WriteImage(data_itk, str(file_path))

def read_nifti(path):
    sitk_img = sitk.ReadImage(path)
    orig_size = np.array(sitk_img.GetSize())
    orig_spacing = np.array(sitk_img.GetSpacing())

    if not all(s == orig_spacing[0] for s in orig_spacing):  # data needs to be isotropic
        logger.warning("Image spacing is anisotropic.")

        if len(set(orig_spacing)) != len(orig_spacing):
            duplicate = [s for s in orig_spacing if list(orig_spacing).count(s) == 2][0]
            target_spacing = [duplicate, duplicate, duplicate]
        else:
            target_spacing = [min(orig_spacing)] * 3
        target_size = np.round(orig_size * (orig_spacing / target_spacing)).astype(int)
        logger.info(f"Adjust spacing from {orig_spacing} to {target_spacing}")

        resampler = sitk.ResampleImageFilter()
        interpolator = sitk.sitkLinear
        resampler.SetInterpolator(interpolator)
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize([int(s) for s in target_size])
        resampler.SetOutputOrigin(sitk_img.GetOrigin())
        resampler.SetOutputDirection(sitk_img.GetDirection())

        # perform resampling
        sitk_img = resampler.Execute(sitk_img)
    else:
        target_spacing = list(orig_spacing)

    img = sitk.GetArrayFromImage(sitk_img)
    return img, target_spacing

def get_paths(cfg):
    image_paths = list(Path(cfg.image_path).iterdir())
    if cfg.mask_path:
        mask_paths = [Path(cfg.mask_path) / f"{p.name}" for p in image_paths]
        assert all(
            mask_path.exists() for mask_path in mask_paths
        ), "All mask paths must exist mask name has to be the same as the image name."
    else:
        mask_paths = None
    return image_paths, mask_paths

def resample(image, factor=None, target_shape=None):
    if factor == 1:
        return image
    
    if target_shape:
        _, _, new_d, new_h, new_w = target_shape
    else:
        _, _, d, h, w = image.shape
        new_d, new_h, new_w = int(round(d / factor)), int(round(h / factor)), int(round(w / factor))
    return F.interpolate(image, size=(new_d, new_h, new_w), mode="trilinear", align_corners=False)

@hydra.main(config_path="configs", config_name="inference", version_base="1.3.2")
def main(cfg):
    # seed libraries
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # set device
    logger.info(f"Using device {cfg.device}.")
    device = cfg.device

    # load model and ckpt
    model = load_model(cfg, device)
    model.to(device)
    model.eval()

    # init pre-processing transforms
    transforms = generate_transforms(cfg.transforms_config)

    # i/o
    output_folder = Path(cfg.output_folder)
    output_folder.mkdir(exist_ok=True)

    image_paths, mask_paths = get_paths(cfg)
    logger.info(f"Found {len(image_paths)} images in {cfg.image_path}.")

    # init sliding window inferer
    logger.debug(f"Sliding window patch size: {cfg.patch_size}")
    logger.debug(f"Sliding window batch size: {cfg.batch_size}.")
    logger.debug(f"Sliding window overlap: {cfg.overlap}.")
    inferer = SlidingWindowInfererAdapt(
        roi_size=cfg.patch_size, sw_batch_size=cfg.batch_size, overlap=cfg.overlap, 
        mode=cfg.mode, sigma_scale=cfg.sigma_scale, padding_mode=cfg.padding_mode
    )

    # loop over images
    metrics_dict = {}
    with torch.no_grad():
        for idx, image_path in tqdm(enumerate(image_paths), total=len(image_paths), desc="Processing images."):
            if image_path.suffix != '.gz':
                raise ValueError(f"Unexpected file extension '{image_path.suffix}' for {image_path.name}. Expected '.nii.gz'.")

            preds = [] # average over test time augmentations
            for scale in cfg.tta.scales:
                # apply pre-processing transforms
                image, spacing = read_nifti(image_path)
                image = transforms(image.astype(np.float32))[None].to(device)
                mask = torch.tensor(read_nifti(mask_paths[idx])[0]).bool() if mask_paths else None
  
                # apply test time augmentation
                if cfg.tta.invert:
                    image = 1 - image if image.mean() > cfg.tta.invert_mean_thresh else image
                    
                if cfg.tta.equalize_hist:
                    image_np = image.cpu().squeeze().numpy()
                    image_equal_hist_np = equalize_hist(image_np, nbins=cfg.tta.hist_bins)
                    image = torch.from_numpy(image_equal_hist_np).to(image.device)[None][None]

                original_shape = image.shape
                image = resample(image, factor=scale)
                logits = inferer(image, model)
                logits = resample(logits, target_shape=original_shape)
                preds.append(logits.cpu().squeeze())

            # merging
            if cfg.merging.max:
                pred = torch.stack(preds).max(dim=0)[0].sigmoid()
            else:
                pred = torch.stack(preds).mean(dim=0).sigmoid()
            pred_thresh = (pred > cfg.merging.threshold).numpy()

            # post-processing
            if cfg.post.apply:
                pred_thresh = remove_small_objects(
                    pred_thresh, min_size=cfg.post.small_objects_min_size, connectivity=cfg.post.small_objects_connectivity
                )

            # save final pred
            write_nifti(
                pred_thresh.astype(np.uint8), 
                output_folder / f"{image_path.name.split('.')[0]}_{cfg.file_app}pred.nii.gz",
                spacing
            )

            write_nifti(
                image.astype(np.float32).squeeze(), 
                output_folder / f"{image_path.name.split('.')[0]}_{cfg.file_app}img.nii.gz",
                spacing
            )

            if mask_paths is not None:
                metrics = Evaluator().estimate_metrics(pred, mask, threshold=cfg.merging.threshold) # no post-processing
                logger.info(f"Dice of {image_path.name.split('.')[0]}: {metrics['dice'].item()}")
                logger.info(f"clDice of {image_path.name.split('.')[0]}: {metrics['cldice'].item()}")
                metrics_dict[image_path.name.split('.')[0]] = metrics

    if mask_paths is not None:
        mean_metrics = calculate_mean_metrics(list(metrics_dict.values()), round_to=cfg.round_to)
        logger.info(f"Mean metrics: dice {mean_metrics['dice'].item()}, cldice {mean_metrics['cldice'].item()}")
    logger.info("Done.")


if __name__ == "__main__":
    main()