import SimpleITK as sitk
import os
import numpy as np
from .utils import save_array, save_metadata, calculate_metadata, convert_sitk_image


def convert_SMH(input_folder: str = "tests/allMRT1", output_dir: str = "data/converted_real/SMH"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labelsTr"), exist_ok=True)

    image_dir = os.path.join(input_folder, "04_FILTERED") if len(os.listdir(os.path.join(input_folder, "04_FILTERED"))) > 0 else os.path.join(input_folder, "03_RESAMPLED")
    mask_dir = os.path.join(input_folder, "06_VESSELS_MASKED")

    print(f"Converting Images...")
    for sample in os.listdir(image_dir):
        if not sample.endswith(".nii.gz"):
            continue
        print(f"Converting {sample}...")
        image = sitk.ReadImage(os.path.join(image_dir, sample))
        array, metadata = convert_sitk_image(image)
        array = array.astype(np.float32)
        metadata = metadata | calculate_metadata(array)
        sample_name = sample.split(".")[0]
        save_array(array, os.path.join(output_dir, "imagesTr", sample_name))
        save_metadata(metadata, os.path.join(output_dir, "imagesTr", sample_name))

    print(f"Converting Masks...")
    for sample in os.listdir(mask_dir):
        print(f"Converting {sample}...")
        mask = sitk.ReadImage(os.path.join(mask_dir, sample))
        array, metadata = convert_sitk_image(mask)
        array = array > 0
        sample_name = sample.split(".")[0]
        save_array(array, os.path.join(output_dir, "labelsTr", sample_name))
        save_metadata(metadata, os.path.join(output_dir, "labelsTr", sample_name))
