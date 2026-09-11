#!/usr/bin/env python3
"""
smooth_mask.py
--------------
Smooth the surface of a binary NIfTI mask using Gaussian filtering,
then re-threshold to recover a binary mask with smoothed edges.

The approach preserves the overall volume by choosing a threshold that
matches the input mask's volume after smoothing.

Usage:
    python smooth_mask.py mask.nii.gz --output mask_smooth.nii.gz
    python smooth_mask.py mask.nii.gz --sigma 1.0 --output mask_smooth.nii.gz
    python smooth_mask.py mask.nii.gz --sigma 1.0 --threshold 0.5 --output mask_smooth.nii.gz

Requirements:
    pip install antspyx
"""

import argparse
import numpy as np
import ants


def smooth_mask(
    mask_path: str,
    output_path: str,
    sigma: float = 1.0,
    threshold: float = 0.5,
    preserve_volume: bool = True,
) -> ants.ANTsImage:
    """
    Smooth a binary mask surface using Gaussian filtering.

    Steps
    -----
    1. Load the binary mask.
    2. Apply Gaussian smoothing (ants.smooth_image) — this blurs the 0/1
       boundary into a continuous [0, 1] probability-like field.
    3. Re-threshold the smoothed image back to binary:
       - If preserve_volume=True  : find the threshold that best conserves
                                    the input non-zero voxel count.
       - If preserve_volume=False : use the fixed --threshold value (default 0.5).

    Parameters
    ----------
    mask_path      : path to input NIfTI mask (.nii / .nii.gz)
    output_path    : path for the smoothed output mask
    sigma          : Gaussian sigma in mm (default: 1.0)
    threshold      : fixed re-binarisation threshold when preserve_volume=False
                     (default: 0.5)
    preserve_volume: if True, choose threshold to match input voxel count

    Returns
    -------
    The smoothed binary ANTsImage.
    """
    # --- Load ---
    print(f"Loading mask : {mask_path}")
    mask = ants.image_read(mask_path)
    print(f"  Shape   : {mask.shape}")
    print(f"  Spacing : {tuple(round(s, 4) for s in mask.spacing)} mm")

    n_in = int((mask.numpy() > 0).sum())
    print(f"  Non-zero voxels : {n_in:,}")

    # --- Gaussian smoothing ---
    print(f"\nApplying Gaussian smoothing (sigma = {sigma} mm)...")
    smoothed = ants.smooth_image(mask, sigma=sigma, sigma_in_physical_coordinates=True)

    # --- Re-threshold ---
    data = smoothed.numpy()

    if preserve_volume:
        print("  Finding threshold that conserves input volume...")
        # Binary-search for the threshold that gives the closest voxel count
        lo, hi = float(data.min()), float(data.max())
        for _ in range(50):          # 50 bisections → precision < 1e-13
            mid = (lo + hi) / 2.0
            if int((data > mid).sum()) > n_in:
                lo = mid
            else:
                hi = mid
        threshold_used = (lo + hi) / 2.0
        print(f"  Threshold used  : {threshold_used:.6f}")
    else:
        threshold_used = threshold
        print(f"  Threshold used  : {threshold_used:.6f} (fixed)")

    smoothed_binary = ants.threshold_image(smoothed, low_thresh=threshold_used, high_thresh=data.max())

    # --- Stats ---
    n_out = int((smoothed_binary.numpy() > 0).sum())
    print(f"\n  Non-zero voxels : {n_in:,}  →  {n_out:,}  (diff: {n_out - n_in:+,})")

    # --- Save ---
    ants.image_write(smoothed_binary, output_path)
    print(f"\nSaved : {output_path}")
    return smoothed_binary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Smooth a binary NIfTI mask surface using Gaussian filtering. "
            "The mask is smoothed then re-thresholded back to binary, "
            "optionally preserving the input voxel count."
        )
    )
    parser.add_argument("mask", help="Input NIfTI mask (.nii / .nii.gz)")
    parser.add_argument(
        "--output", "-o",
        default="mask_smooth.nii.gz",
        help="Output file path (default: mask_smooth.nii.gz)",
    )
    parser.add_argument(
        "--sigma", "-s",
        type=float,
        default=1.0,
        help="Gaussian sigma in mm — controls smoothing strength (default: 1.0)",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help=(
            "Re-binarisation threshold applied to the smoothed image "
            "(default: 0.5). Only used when --no-preserve-volume is set."
        ),
    )
    parser.add_argument(
        "--no-preserve-volume",
        action="store_true",
        default=False,
        help=(
            "Use the fixed --threshold instead of auto-finding the threshold "
            "that conserves the input voxel count."
        ),
    )
    args = parser.parse_args()

    smooth_mask(
        mask_path=args.mask,
        output_path=args.output,
        sigma=args.sigma,
        threshold=args.threshold,
        preserve_volume=not args.no_preserve_volume,
    )


if __name__ == "__main__":
    main()