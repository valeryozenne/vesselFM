#!/usr/bin/env python3
"""
binary_opening.py
-----------------
Perform binary morphological opening (erosion followed by dilation) on a
NIfTI mask using ANTsPy's iMath.

Usage:
    python binary_opening.py mask.nii.gz --output mask_opened.nii.gz
    python binary_opening.py mask.nii.gz --radius 2 --output mask_opened.nii.gz

Requirements:
    pip install antspyx
"""

import argparse
import ants


def binary_opening(
    mask_path: str,
    output_path: str,
    radius: int = 1,
) -> ants.ANTsImage:
    """
    Apply binary morphological opening to a NIfTI mask.

    Opening = erosion (ME) followed by dilation (MD).
    This removes small isolated voxels and smooths boundaries
    while preserving the overall shape.

    Parameters
    ----------
    mask_path   : path to input NIfTI mask (.nii / .nii.gz)
    output_path : path for the output mask
    radius      : morphological kernel radius in voxels (default: 1)

    Returns
    -------
    The opened ANTsImage.
    """
    # --- Load ---
    print(f"Loading mask : {mask_path}")
    mask = ants.image_read(mask_path)
    print(f"  Shape   : {mask.shape}")
    print(f"  Spacing : {tuple(round(s, 4) for s in mask.spacing)} mm")

    # --- Opening: erosion then dilation ---
    print(f"\nApplying binary opening (radius = {radius} voxel(s))...")
    eroded  = ants.iMath(mask, "ME", radius)   # morphological erosion
    opened  = ants.iMath(eroded, "MD", radius) # morphological dilation

    # --- Stats ---
    n_in  = int((mask.numpy()   > 0).sum())
    n_out = int((opened.numpy() > 0).sum())
    print(f"  Non-zero voxels : {n_in:,}  →  {n_out:,}  (diff: {n_out - n_in:+,})")

    # --- Save ---
    ants.image_write(opened, output_path)
    print(f"\nSaved : {output_path}")
    return opened


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Binary morphological opening on a NIfTI mask using ANTsPy. "
            "Opening = erosion then dilation with the same radius, which "
            "removes small spurious voxels while preserving overall shape."
        )
    )
    parser.add_argument("mask", help="Input NIfTI mask (.nii / .nii.gz)")
    parser.add_argument(
        "--output", "-o",
        default="mask_opened.nii.gz",
        help="Output file path (default: mask_opened.nii.gz)",
    )
    parser.add_argument(
        "--radius", "-r",
        type=int,
        default=1,
        help="Morphological kernel radius in voxels (default: 1)",
    )
    args = parser.parse_args()

    binary_opening(
        mask_path=args.mask,
        output_path=args.output,
        radius=args.radius,
    )


if __name__ == "__main__":
    main()