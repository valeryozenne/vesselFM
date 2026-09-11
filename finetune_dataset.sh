mkdir -p data/finetune/octa/train/0 data/finetune/octa/train/1 data/finetune/octa/train/2 data/finetune/octa/val/0 data/finetune/octa/test/0 data/finetune/octa/test/1
mkdir -p data/finetune/msd/train/0 data/finetune/msd/train/1 data/finetune/msd/train/2 data/finetune/msd/val/0 
for i in {0..295}; do
    mkdir -p data/finetune/msd/test/$i
done
mkdir -p data/finetune/bvem/train/0 data/finetune/bvem/train/1 data/finetune/bvem/train/2 data/finetune/bvem/val/0 data/finetune/bvem/test/0 data/finetune/bvem/test/1

cp data/d_drand/foreground/manual_annotations/test_set_m4_0.nii        data/finetune/octa/val/0/img.nii
cp data/d_drand/foreground/manual_annotations/test_set_m4_0_label.nii  data/finetune/octa/val/0/mask.nii

cp data/d_drand/foreground/manual_annotations/test_set_m4_1.nii        data/finetune/octa/train/0/img.nii
cp data/d_drand/foreground/manual_annotations/test_set_m4_1_label.nii  data/finetune/octa/train/0/mask.nii
cp data/d_drand/foreground/manual_annotations/test_set_m44_0.nii       data/finetune/octa/train/1/img.nii
cp data/d_drand/foreground/manual_annotations/test_set_m44_0_label.nii data/finetune/octa/train/1/mask.nii
cp data/d_drand/foreground/manual_annotations/test_set_m44_1.nii       data/finetune/octa/train/2/img.nii
cp data/d_drand/foreground/manual_annotations/test_set_m44_1_label.nii data/finetune/octa/train/2/mask.nii

cp data/d_drand/foreground/manual_annotations/test_set_m78_0.nii       data/finetune/octa/test/0/img.nii
cp data/d_drand/foreground/manual_annotations/test_set_m78_0_label.nii data/finetune/octa/test/0/mask.nii
cp data/d_drand/foreground/manual_annotations/test_set_m78_1.nii       data/finetune/octa/test/1/img.nii
cp data/d_drand/foreground/manual_annotations/test_set_m78_1_label.nii data/finetune/octa/test/1/mask.nii

# cp ~/Téléchargements/msd_task8/imagesTr/hepaticvessel_001.nii.gz  data/finetune/msd/val/0/img.nii.gz
# cp ~/Téléchargements/msd_task8/labelsTr/hepaticvessel_001.nii.gz  data/finetune/msd/val/0/mask.nii.gz

# cp ~/Téléchargements/msd_task8/imagesTr/hepaticvessel_002.nii.gz  data/finetune/msd/train/0/img.nii.gz
# cp ~/Téléchargements/msd_task8/labelsTr/hepaticvessel_002.nii.gz  data/finetune/msd/train/0/mask.nii.gz
# cp ~/Téléchargements/msd_task8/imagesTr/hepaticvessel_004.nii.gz  data/finetune/msd/train/1/img.nii.gz
# cp ~/Téléchargements/msd_task8/labelsTr/hepaticvessel_004.nii.gz  data/finetune/msd/train/1/mask.nii.gz
# cp ~/Téléchargements/msd_task8/imagesTr/hepaticvessel_005.nii.gz  data/finetune/msd/train/2/img.nii.gz
# cp ~/Téléchargements/msd_task8/labelsTr/hepaticvessel_005.nii.gz  data/finetune/msd/train/2/mask.nii.gz

# offset=0
# for i in {0..255}; do
#     # si le fichier n'existe pas, on prend le suivant mais on garde le bon numéro d'image pour le nommage
#     while [ ! -f ~/Téléchargements/msd_task8/imagesTr/hepaticvessel_$(printf "%03d" $((i+6+$offset))).nii.gz ]; do
#         echo "File ~/Téléchargements/msd_task8/imagesTr/hepaticvessel_$(printf "%03d" $((i+6+$offset))).nii.gz does not exist, skipping to next file"
#         offset=$offset+1
#     done
#     cp ~/Téléchargements/msd_task8/imagesTr/hepaticvessel_$(printf "%03d" $((i+6+$offset))).nii.gz  data/finetune/msd/test/$i/img.nii.gz
#     cp ~/Téléchargements/msd_task8/labelsTr/hepaticvessel_$(printf "%03d" $((i+6+$offset))).nii.gz  data/finetune/msd/test/$i/mask.nii.gz
# done

cp data/d_real/MSD_Task8/12/img.npy  data/finetune/msd/val/0/img.npy
cp data/d_real/MSD_Task8/12/mask.npy  data/finetune/msd/val/0/mask.npy

cp data/d_real/MSD_Task8/1/img.npy  data/finetune/msd/train/0/img.npy
cp data/d_real/MSD_Task8/1/mask.npy  data/finetune/msd/train/0/mask.npy
cp data/d_real/MSD_Task8/5/img.npy  data/finetune/msd/train/1/img.npy
cp data/d_real/MSD_Task8/5/mask.npy  data/finetune/msd/train/1/mask.npy
cp data/d_real/MSD_Task8/6/img.npy  data/finetune/msd/train/2/img.npy
cp data/d_real/MSD_Task8/6/mask.npy  data/finetune/msd/train/2/mask.npy

offset=15
for i in {0..295}; do
    cp data/d_real/MSD_Task8/$((i+$offset))/img.npy  data/finetune/msd/test/$i/img.npy
    cp data/d_real/MSD_Task8/$((i+$offset))/mask.npy  data/finetune/msd/test/$i/mask.npy
done

# finetune des données smartheat
mkdir -p data/finetune/smh/train/0 data/finetune/smh/val/0 data/finetune/smh/test/0 data/finetune/smh/test/1 data/finetune/smh/test/2

cp tests/MWA20220519a/VESSELS_BESTY_MASKED/016__fl3d2-t1-vibe-dixon-tra-p4-bh-320-w.nii.gz data/finetune/smh/train/0/mask.nii.gz
cp tests/MWA20220519a/RESAMPLED/016__fl3d2-t1-vibe-dixon-tra-p4-bh-320-w.nii.gz data/finetune/smh/train/0/img.nii.gz

cp tests/MWA20220519a/VESSELS_BESTY_MASKED/011__fl3d2-t1-vibe-dixon-tra-p4-bh-320-w.nii.gz data/finetune/smh/val/0/mask.nii.gz
cp tests/MWA20220519a/RESAMPLED/011__fl3d2-t1-vibe-dixon-tra-p4-bh-320-w.nii.gz data/finetune/smh/val/0/img.nii.gz

cp tests/MWA20220519a/VESSELS_BESTY_MASKED/002__fl3d2-t1-vibe-dixon-tra-p4-bh-320-w.nii.gz data/finetune/smh/test/0/mask.nii.gz
cp tests/MWA20220519a/RESAMPLED/002__fl3d2-t1-vibe-dixon-tra-p4-bh-320-w.nii.gz data/finetune/smh/test/0/img.nii.gz
cp tests/MWA20220920a/VESSELS/014__fl3d2-t1-vibe-dixon-tra-p4-bh-320-wpred.nii.gz data/finetune/smh/test/1/mask.nii.gz
cp tests/MWA20220920a/RAW-NIFTI/014__fl3d2-t1-vibe-dixon-tra-p4-bh-320-w.nii.gz data/finetune/smh/test/1/img.nii.gz
cp tests/MWA20220920a/VESSELS/015__fl3d2-t1-vibe-dixon-cor-p4-bh-320-wpred.nii.gz data/finetune/smh/test/2/mask.nii.gz
cp tests/MWA20220920a/RAW-NIFTI/015__fl3d2-t1-vibe-dixon-cor-p4-bh-320-w.nii.gz data/finetune/smh/test/2/img.nii.gz