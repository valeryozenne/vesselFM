import logging
from typing import Tuple
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from vesselfm.seg.utils.io import determine_reader_writer
from vesselfm.seg.utils.data import generate_transforms

logger = logging.getLogger(__name__)


class UnionDataset(Dataset):
    def __init__(self, dataset_configs, mode, finetune=False):
        super().__init__()
        self.finetune = finetune
        self.datasets, probs = [], []
        self.len = 0
        for name, dataset_config in dataset_configs.items():
            data_dir = Path(dataset_config.path) / mode if finetune else Path(dataset_config.path)
    
            # --- 1. PRÉ-RÉSOLUTION ET FILTRAGE DES CHEMINS ---
            valid_samples = []
            for sample_dir in sorted(list(data_dir.iterdir())):
                # Trouver les fichiers une seule fois au démarrage
                img_paths = list(sample_dir.glob('*img*'))
                mask_paths = list(sample_dir.glob('*mask*'))
                
                if not img_paths or not mask_paths:
                    continue
                    
                img_path = img_paths[0]
                mask_path = mask_paths[0]

                # Filtrer directement ici pour éviter la boucle infinie dans __getitem__
                if dataset_config.filter_dataset_IDs is not None:
                    if int(img_path.stem.split("_")[-1]) in dataset_config.filter_dataset_IDs:
                        continue 
                        
                valid_samples.append({
                    "img_path": img_path,
                    "mask_path": mask_path
                })

            self.len += len(valid_samples)
            self.datasets.append(
                {
                    "name": name,
                    "samples": valid_samples,
                    # --- 2. STOCKER LA CLASSE DU READER (SANS L'INSTANCIER AVEC ()) ---
                    "reader_cls": determine_reader_writer(dataset_config.file_format),
                    "transforms": generate_transforms(dataset_config.transforms[mode]),
                }
            )
            probs.append(dataset_config.sample_prop)

        probs = torch.tensor(probs, dtype=torch.float32)
        if (probs < 0).any():
            raise ValueError("Erreur : La configuration contient un sample_prop négatif.")
            
        probs_sum = probs.sum()
        if probs_sum == 0:
            logger.warning("La somme des sample_prop est de 0. Utilisation d'une distribution uniforme.")
            self.probs = torch.ones_like(probs) / len(probs)
        else:
            self.probs = probs / probs_sum

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Sélection du dataset
        dataset_id = torch.multinomial(self.probs, 1).item()
        dataset = self.datasets[dataset_id]

        # Sélection de l'échantillon (plus de boucle while True infinie)
        data_idx = idx if self.finetune else torch.randint(0, len(dataset["samples"]), (1,)).item()
        sample = dataset["samples"][data_idx]
        
        img_path = sample["img_path"]
        mask_path = sample["mask_path"]

        # --- 3. INSTANCIATION DU READER DANS LE WORKER ---
        reader = dataset["reader_cls"]()

        img = reader.read_images(str(img_path))[0].astype(np.float32)
        mask = reader.read_images(str(mask_path))[0].astype(bool)

        if np.isnan(img).any() or np.isnan(mask).any():
            print(f"ATTENTION: Valeurs NaN détectées dans {img_path}")

        if img.size == 0 or 0 in img.shape:
            raise ValueError(f"CRASH : L'image chargée est vide ! Fichier : {img_path}")

        data_dict = {'Image': img, 'Mask': mask}
        is_empty_foreground = not mask.any() or not img.any()

        for t in dataset['transforms'].transforms:
            if is_empty_foreground and "CropForegroundd" in t.__class__.__name__:
                continue 
            data_dict = t(data_dict)

        return data_dict['Image'], data_dict['Mask'] > 0