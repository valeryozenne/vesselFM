# script pour écarter les données de 2026 vers un autre dossier

import os
import shutil 
import pathlib
import argparse

parser = argparse.ArgumentParser(description="Move 2026 data to a new directory")
parser.add_argument('--converted_data_dir', type=str, default="data/converted_real/allMRT1", help="Path to the converted data directory")
parser.add_argument('--new_data_dir', type=str, default="data/converted_real/smh26", help="Path to the new data directory")
args = parser.parse_args()

converted_data_dir = args.converted_data_dir
new_data_dir = args.new_data_dir

pathlib.Path(new_data_dir).mkdir(parents=True, exist_ok=True)
# dossiers imagesTr et labelsTr
for subdir in ["imagesTr", "labelsTr"]:
    old_subdir = os.path.join(converted_data_dir, subdir)
    new_subdir = os.path.join(new_data_dir, subdir)
    pathlib.Path(new_subdir).mkdir(parents=True, exist_ok=True)

    # move files from the old directory to the new one
    for dirnames in os.listdir(old_subdir):
        print(f"Processing {dirnames}...")
        if dirnames.startswith("MWA2021") or dirnames.startswith("MWA2022") or dirnames.startswith("MWA2023") or dirnames.startswith("MWA2024") or dirnames.startswith("MWA2025"):
            continue  # skip these directories
        # move the directory to the new location
        shutil.move(os.path.join(old_subdir, dirnames), os.path.join(new_subdir, dirnames))