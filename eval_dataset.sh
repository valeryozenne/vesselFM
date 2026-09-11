# mv data/d_real/MiniVess/0/mask.nii data/evaluation/minivess_mask.nii
# mv data/d_real/MiniVess/0/img.nii data/evaluation/minivess.nii
# mv data/d_real/DeepVess/0/mask.nii data/evaluation/deepvess_mask.nii
# mv data/d_real/DeepVess/0/img.nii data/evaluation/deepvess.nii

cp data/d_real/BvEM/0/img.npy data/evaluation/images/bvem.npy
cp data/d_real/BvEM/0/mask.npy data/evaluation/masks/bvem.npy

cp data/d_real/MSD_Task8/0/img.npy data/evaluation/images/msd.npy
cp data/d_real/MSD_Task8/0/mask.npy data/evaluation/masks/msd.npy

cp data/d_real/OCTA_no_rot/0/img.npy data/evaluation/images/octa.npy
cp data/d_real/OCTA_no_rot/0/mask.npy data/evaluation/masks/octa.npy
