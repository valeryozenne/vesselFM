import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

def find_valid_coordinates(img_dir, lbl_dir, z_start, patch_size, num_needed, split_name, y_min, y_max, x_min, x_max):
    """
    Scanne le volume pour trouver des coordonnées (y, x) qui ne contiennent AUCUNE slice corrompue.
    """
    print(f"Recherche de {num_needed} patchs sains pour '{split_name}' (Slices {z_start} à {z_start+patch_size-1})...")
    step = patch_size + 50 # Buffer de 50 pixels pour garantir qu'il n'y a aucun chevauchement entre les patchs
    
    # Création d'une grille de candidats possibles
    candidates = [(y, x) for y in range(y_min, y_max - patch_size + 1, step)
                         for x in range(x_min, x_max - patch_size + 1, step)]
    
    valid_candidates = set(candidates)

    for z in tqdm(range(z_start, z_start + patch_size), desc=f"Scan de {split_name}"):
        if len(valid_candidates) < num_needed:
            raise ValueError(f"Échec: Trop d'artefacts, impossible de trouver {num_needed} zones saines pour {split_name}.")

        img_slice = np.load(img_dir / f"{z:04d}.npy")
        lbl_slice = np.load(lbl_dir / f"{z:04d}.npy")

        to_remove = []
        for (y, x) in valid_candidates:
            img_crop = img_slice[y:y+patch_size, x:x+patch_size]
            lbl_crop = lbl_slice[y:y+patch_size, x:x+patch_size]

            # DÉTECTEUR 1 : Image corrompue ou padding (Zone parfaitement lisse/grise)
            if img_crop.std() < 3.0:
                to_remove.append((y, x))
                continue

            # DÉTECTEUR 2 : Label corrompu
            # On cherche si la slice est uniformément remplie par une valeur aberrante (ex: 128, 255).
            # Note: on accepte "0" car une slice 128x128 peut légitimement ne croiser aucun vaisseau.
            unique_lbls = np.unique(lbl_crop)
            if len(unique_lbls) == 1 and unique_lbls[0] != 0:
                to_remove.append((y, x))
                continue

        # Élimination des mauvais candidats
        for c in to_remove:
            if c in valid_candidates:
                valid_candidates.remove(c)

    final_coords = list(valid_candidates)[:num_needed]
    print(f"-> Succès ! Coordonnées validées pour {split_name} : {final_coords}\n")
    return final_coords

def extract_and_save_patches(img_dir, lbl_dir, out_base_dir, split_name, z_start, patch_size, coords, patch_ids):
    pz, py, px = patch_size, patch_size, patch_size
    num_patches = len(coords)
    
    sample_img = np.load(img_dir / f"{z_start:04d}.npy", mmap_mode='r')
    sample_lbl = np.load(lbl_dir / f"{z_start:04d}.npy", mmap_mode='r')
    
    img_patches = [np.zeros((pz, py, px), dtype=sample_img.dtype) for _ in range(num_patches)]
    lbl_patches = [np.zeros((pz, py, px), dtype=sample_lbl.dtype) for _ in range(num_patches)]
    
    print(f"Extraction finale pour '{split_name}' aux coordonnées {coords}...")
    for z in tqdm(range(pz), desc=f"Génération {split_name}"):
        slice_idx = z_start + z
        img_slice = np.load(img_dir / f"{slice_idx:04d}.npy")
        lbl_slice = np.load(lbl_dir / f"{slice_idx:04d}.npy")
        
        for i, (y, x) in enumerate(coords):
            img_patches[i][z] = img_slice[y:y+py, x:x+px]
            lbl_patches[i][z] = lbl_slice[y:y+py, x:x+px]
            
    for i, p_id in enumerate(patch_ids):
        out_dir = out_base_dir / split_name / str(p_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "img.npy", img_patches[i])
        np.save(out_dir / "mask.npy", lbl_patches[i])
    print(f"-> Fichiers sauvegardés pour {split_name} !\n")

def main():
    base_data_dir = Path("data/converted_real/bvem")
    img_dir = base_data_dir / "imagesTr" / "mouse_microns-phase2_256-320nm_crop"
    lbl_dir = base_data_dir / "labelsTr" / "mouse_microns-phase2_256-320nm_crop"
    out_dir = Path("data/finetune/bvem")
    
    # 1. Scanner toute la MOITIÉ GAUCHE de l'image pour Train et Val (x de 200 à 2500)
    tv_coords = find_valid_coordinates(
        img_dir, lbl_dir, z_start=2, patch_size=128, num_needed=4, 
        split_name="Train_Val", y_min=400, y_max=3300, x_min=400, x_max=2500
    )
    train_coords = tv_coords[:3]
    val_coords = [tv_coords[3]]

    # 2. Scanner toute la MOITIÉ DROITE de l'image pour Test (x de 2600 à 4600)
    # Le chevauchement XY avec Train/Val est donc mathématiquement impossible.
    test_coords = find_valid_coordinates(
        img_dir, lbl_dir, z_start=250, patch_size=500, num_needed=3, 
        split_name="Test", y_min=200, y_max=2900, x_min=2600, x_max=4600
    )

    # 3. Extraction
    extract_and_save_patches(img_dir, lbl_dir, out_dir, "train", z_start=2, patch_size=128, coords=train_coords, patch_ids=[0, 1, 2])
    extract_and_save_patches(img_dir, lbl_dir, out_dir, "val", z_start=2, patch_size=128, coords=val_coords, patch_ids=[0])
    extract_and_save_patches(img_dir, lbl_dir, out_dir, "test", z_start=250, patch_size=500, coords=test_coords, patch_ids=[0, 1, 2])

    print("Pipeline de découpage terminé avec succès ! Plus aucun patch corrompu.")

if __name__ == "__main__":
    main()