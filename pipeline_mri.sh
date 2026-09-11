DIR="tests/MWA20220519a/vesselfm_all_pt"

mkdir -p $DIR/00_LIVER $DIR/04_FILTERED $DIR/02_CROPPED $DIR/03_RESAMPLED $DIR/05_VESSELS $DIR/06_VESSELS_MASKED $DIR/01_LIVER_REGRID

RAWNIFTIS=$(ls $DIR/RAW-NIFTI/*.nii.gz)
RAWNIFTIBASENAMES=$(ls $DIR/RAW-NIFTI/*.nii.gz | xargs -n 1 basename | sed 's/.nii.gz//')
echo "Found $RAWNIFTIBASENAMES"

# LIVER SEGMENTATION
# activation de l'environnement qui contient TotalSegmentator
source /workspace_QMRI/USERS_CODE/hsalles/00_venvs/smart-heat-venv/bin/activate

for RAWNIFTI in $RAWNIFTIS; do
    BASENAME=$(basename $RAWNIFTI .nii.gz)

    # LIVER SEGMENTATION WITH TOTALSEGMENTATOR

    if [ ! -f ./$DIR/00_LIVER/${BASENAME}.nii.gz ]; then
        echo -e "\033[33m[PIPELINE] Liver segmentation $RAWNIFTI\033[0m"
        TotalSegmentator -i $RAWNIFTI -o ./$DIR/00_LIVER -ta total_mr -rs liver
        # rename the output file from liver.nii.gz to the original filename
        mv ./$DIR/00_LIVER/liver.nii.gz ./$DIR/00_LIVER/${BASENAME}.nii.gz
    fi

    # CROP AROUND THE LIVER

    echo -e "\033[33m[PIPELINE] Cropping $RAWNIFTI\033[0m"
    mrgrid $RAWNIFTI crop ./$DIR/02_CROPPED/${BASENAME}.nii.gz --mask ./$DIR/00_LIVER/${BASENAME}.nii.gz --uniform -5 --force

    # RESAMPLE TO 1x1x1 mm3

    echo -e "\033[33m[PIPELINE] Resampling $RAWNIFTI\033[0m"
    mrgrid ./$DIR/02_CROPPED/${BASENAME}.nii.gz regrid ./$DIR/03_RESAMPLED/${BASENAME}.nii.gz --voxel 1,1,1 --force

    # APPLY N4 BIAS FIELD CORRECTION with ANTS

    echo -e "\033[33m[PIPELINE] Applying N4 bias field correction to $RAWNIFTI\033[0m"
    N4BiasFieldCorrection \
        -i ./$DIR/03_RESAMPLED/${BASENAME}.nii.gz \
        -s 10 \
        -o ./$DIR/04_FILTERED/${BASENAME}.nii.gz

done

# INFERENCE WITH VESSELFM

source /workspace_QMRI/USERS_CODE/hsalles/00_venvs/vesselFM_env/bin/activate

python vesselfm/seg/inference.py

for RAWNIFTI in $RAWNIFTIS; do
    BASENAME=$(basename $RAWNIFTI .nii.gz)
    FILENAME=$(ls ./$DIR/05_VESSELS/${BASENAME}*.nii.gz)

    mrgrid ./$DIR/00_LIVER/${BASENAME}.nii.gz regrid --template $FILENAME ./$DIR/01_LIVER_REGRID/${BASENAME}.nii.gz --interp nearest --force

    mrcalc $FILENAME ./$DIR/01_LIVER_REGRID/${BASENAME}.nii.gz --mult ./$DIR/06_VESSELS_MASKED/${BASENAME}.nii.gz --datatype uint8 --force
done