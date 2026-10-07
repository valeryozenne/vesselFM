#!/usr/bin/env python3
"""
resample_mask.py
----------------
Resample a NIfTI binary mask to a given isotropic (or anisotropic) resolution
using ANTsPy with nearest-neighbour interpolation to preserve binary values.

Usage:
    # 1 mm isotropic (default)
    python resample_mask.py mask.nii.gz --output mask_1mm.nii.gz

    # Custom isotropic resolution
    python resample_mask.py mask.nii.gz --resolution 2.0 --output mask_2mm.nii.gz

    # Anisotropic resolution (x y z)
    python resample_mask.py mask.nii.gz --resolution 1.0 1.0 2.0 --output mask_aniso.nii.gz

    # Resample to the grid of a reference image
    python resample_mask.py mask.nii.gz --reference reference.nii.gz --output mask_resampled.nii.gz

Requirements:
    pip install antspyx
"""

import argparse
from typing import Optional, Tuple
import ants


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def resample_mask(
    mask_path: str,
    output_path: str,
    resolution: Optional[Tuple[float, ...]] = None,
    reference_path: Optional[str] = None,
) -> ants.ANTsImage:
    """
    Load a NIfTI mask, resample it, and save the result.

    Parameters
    ----------
    mask_path      : path to the input mask (.nii / .nii.gz)
    output_path    : path for the resampled output
    resolution     : target voxel spacing in mm, e.g. (1.0,) for isotropic or
                     (1.0, 1.0, 2.0) for anisotropic. Ignored if reference_path is set.
    reference_path : if provided, resample onto this image's grid instead.

    Returns
    -------
    The resampled ANTsImage.
    """  
    # --- Load ---
    print(f"Loading mask : {mask_path}")
    mask = ants.image_read(mask_path)
    print(f"  Input  shape   : {mask.shape}")
    print(f"  Input  spacing : {tuple(round(s, 4) for s in mask.spacing)} mm")

    # --- Resample ---
    if reference_path:
        print(f"\nResampling to reference grid : {reference_path}")
        ref = ants.image_read(reference_path)
        resampled = ants.resample_image_to_target(
            mask, ref, interp_type="nearestNeighbor"
        )
    else:
        ndim = mask.dimension

        # Build target spacing tuple
        if resolution is None:
            target_spacing = (1.0,) * ndim          # default: 1 mm iso
        elif len(resolution) == 1:
            target_spacing = (resolution[0],) * ndim  # isotropic
        elif len(resolution) == ndim:
            target_spacing = tuple(resolution)        # anisotropic
        else:
            raise ValueError(
                f"--resolution expects 1 value (isotropic) or {ndim} values "
                f"(one per dimension), got {len(resolution)}."
            )

        print(f"\nTarget spacing : {target_spacing} mm")

        # Skip if already at target resolution
        if tuple(round(s, 4) for s in mask.spacing) == tuple(round(s, 4) for s in target_spacing):
            print("Mask is already at the requested resolution — saving as-is.")
            ants.image_write(mask, output_path)
            return mask

        # interp_type=1 → nearest neighbour (preserves 0/1 binary values)
        resampled = ants.resample_image(
            mask, target_spacing, use_voxels=False, interp_type=1
        )

    print(f"  Output shape   : {resampled.shape}")
    print(f"  Output spacing : {tuple(round(s, 4) for s in resampled.spacing)} mm")

    # Report voxel count change
    n_in  = int((mask.numpy() > 0).sum())
    n_out = int((resampled.numpy() > 0).sum())
    print(f"\n  Non-zero voxels : {n_in:,}  →  {n_out:,}")

    # --- Save ---
    ants.image_write(resampled, output_path)
    print(f"\nSaved : {output_path}")
    return resampled


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Resample a NIfTI mask to a given resolution using ANTsPy "
            "(nearest-neighbour interpolation to preserve binary values)."
        )
    )
    parser.add_argument("mask", help="Input NIfTI mask (.nii / .nii.gz)")
    parser.add_argument(
        "--output", "-o",
        default="mask_resampled.nii.gz",
        help="Output file path (default: mask_resampled.nii.gz)",
    )
    parser.add_argument(
        "--resolution", "-r",
        type=float,
        nargs="+",
        default=None,
        metavar="MM",
        help=(
            "Target resolution in mm. "
            "One value → isotropic (e.g. --resolution 1.0). "
            "Multiple values → one per dimension (e.g. --resolution 1.0 1.0 2.0). "
            "Default: 1.0 mm isotropic. Ignored if --reference is set."
        ),
    )
    parser.add_argument(
        "--reference",
        default=None,
        metavar="PATH",
        help="Resample onto the voxel grid of this NIfTI image instead of --resolution.",
    )
    args = parser.parse_args()

    resample_mask(
        mask_path=args.mask,
        output_path=args.output,
        resolution=args.resolution,
        reference_path=args.reference,
    )


if __name__ == "__main__":
    main()