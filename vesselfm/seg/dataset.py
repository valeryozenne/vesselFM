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
    """
    Dataset that accumulates all given datasets.
    """
    def __init__(self, dataset_configs, mode, finetune=False):
        super().__init__()
        # init datasets
        self.finetune = finetune
        self.datasets, probs = [], []
        self.len = 0
        for name, dataset_config in dataset_configs.items():
            data_dir = Path(dataset_config.path) / mode if finetune else Path(dataset_config.path)
            paths = sorted(list(data_dir.iterdir())) # ensures that we use same 1-shot sample

            self.len += len(paths)
            self.datasets.append(
                {
                    "name": name,
                    "paths": paths,
                    "reader": determine_reader_writer(dataset_config.file_format)(),
                    "transforms": generate_transforms(dataset_config.transforms[mode]),
                    "sample_prop": dataset_config.sample_prop,
                    "filter_dataset_IDs": dataset_config.filter_dataset_IDs
                }
            )
            probs.append(dataset_config.sample_prop)

        # ensure that probs sum up to 1
        probs = torch.tensor(probs, dtype=torch.float32) # 1. Forcer le type float
        
        # 2. Vérifier qu'il n'y a pas de valeurs négatives
        if (probs < 0).any():
            raise ValueError("Erreur : La configuration contient un sample_prop négatif.")
            
        probs_sum = probs.sum()
        
        # 3. Sécuriser la division par zéro
        if probs_sum == 0:
            logger.warning("La somme des sample_prop est de 0. Utilisation d'une distribution uniforme.")
            self.probs = torch.ones_like(probs) / len(probs)
        else:
            self.probs = probs / probs_sum

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # sample dataset
        dataset_id = torch.multinomial(self.probs, 1).item()
        dataset = self.datasets[dataset_id]

        # sample data sample
        while True:
            data_idx = idx if self.finetune else torch.randint(0, len(dataset["paths"]), (1,)).item()
            sample_id =  dataset["paths"][data_idx]

            img_path = [path for path in sample_id.iterdir() if 'img' in path.name][0]
            mask_path = [path for path in sample_id.iterdir() if 'mask' in path.name][0]

            if dataset['filter_dataset_IDs'] is not None:
                if int(img_path.stem.split("_")[-1]) in dataset['filter_dataset_IDs']:
                    continue

            img = dataset['reader'].read_images(str(img_path))[0].astype(np.float32)
            mask = dataset['reader'].read_images(str(mask_path))[0].astype(bool)

            # Vérifier si l'image ou le masque sont corrompus par des NaN
            if np.isnan(img).any() or np.isnan(mask).any():
                print(f"ATTENTION: Valeurs NaN détectées dans {img_path}")

            # --- DEBUG BLOCK ---
            # Vérifier si l'image est vide dès le chargement
            if img.size == 0 or 0 in img.shape:
                raise ValueError(f"CRASH : L'image chargée est vide ! Fichier : {img_path}")

            # Si l'image n'est pas vide ici, c'est qu'une de vos transformations 
            # (avant ScaleIntensityRangePercentilesd) la rend vide.
            # -------------------

            # transformed = dataset['transforms']({'Image': img, 'Mask': mask})
            data_dict = {'Image': img, 'Mask': mask}

            # Vérifier si le masque est complètement vide (aucun vaisseau)
            # OU si l'image est toute noire
            is_empty_foreground = not mask.any() or not img.any()

            # Appliquer les transformations une par une manuellement
            for t in dataset['transforms'].transforms:
                # Si l'image est vide, on ignore spécifiquement le recadrage pour éviter le crash
                if is_empty_foreground and "CropForegroundd" in t.__class__.__name__:
                    continue 
                    
                data_dict = t(data_dict)

            transformed = data_dict
            return transformed['Image'], transformed['Mask'] > 0
