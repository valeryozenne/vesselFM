import contextlib
import os
from pathlib import Path
import sys

from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile
import webdataset as wds

@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        # sys.stdout = sys.stderr = DummyTqdmFile(orig_out_err[0])
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err

def process_dreal():
    print("Début de conversion domaine expérimental")
    ROOT_DIR = Path("data/d_real")
    os.makedirs("data/wds", exist_ok=True)

    # 1. Collecter tous les échantillons existants
    samples: dict[str, list[tuple[str, str, str]]] = {}
    for dataset_dir in tqdm(sorted(ROOT_DIR.iterdir()), desc=ROOT_DIR.name, position=0):
        if not dataset_dir.is_dir():
            continue
        pattern = "data/wds/" + dataset_dir.name + "-%06d.tar"
        samples[pattern] = []
        for sample_dir in tqdm(sorted(dataset_dir.iterdir()), desc=dataset_dir.name, position=1, leave=False):
            img_path = sample_dir / "img.npy"
            mask_path = sample_dir / "mask.npy"

            if img_path.exists() and mask_path.exists():
                # Clé unique pour éviter les collisions entre sous-datasets
                sample_key = f"{dataset_dir.name}_{sample_dir.name}"
                samples[pattern].append((sample_key, img_path, mask_path))

    print(f"Total de datasets trouvés : {len(samples)}")

    # 2. Écriture en Shards avec ShardWriter
    # maxcount : nb d'échantillons par shard | maxsize : taille max en octets (~10 Go ici)
    with std_out_err_redirect_tqdm() as orig_stdout:
        for pattern, sample in samples.items():
            with wds.ShardWriter(
                pattern, maxcount=1000, maxsize=1024 * 1024 * 1024 * 10
            ) as sink:
                for key, img_path, mask_path in tqdm(sample, file=orig_stdout, dynamic_ncols=True):
                    # Lire les fichiers sous forme d'octets bruts (évite le réencodage)
                    with open(img_path, "rb") as f_img:
                        img_bytes = f_img.read()

                    with open(mask_path, "rb") as f_mask:
                        mask_bytes = f_mask.read()

                    sink.write({
                        "__key__": key,
                        "img.npy": img_bytes,
                        "mask.npy": mask_bytes,
                    })

    print("Conversion terminée !")


def process_drand(dir: Path):
    print("Début de conversion domaine randomisé")
    ROOT_DIR = Path("data/d_drand/d_drand")
    DEST_DIR = dir / "wds_drand"
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    DRAND_SIZE = 500000
    PATTERN = "drand-%06d.tar"

    # 2. Écriture en Shards avec ShardWriter
    # maxcount : nb d'échantillons par shard | maxsize : taille max en octets (~3 Go ici)
    with std_out_err_redirect_tqdm() as orig_stdout:
        with wds.ShardWriter(
            str(DEST_DIR) + '/' + PATTERN, maxcount=160, maxsize=1024 * 1024 * 1024 * 3
        ) as sink:
            for i in tqdm(range(DRAND_SIZE), desc="Consitution des shards", file=orig_stdout, dynamic_ncols=True):
                img_path = ROOT_DIR / str(i) / "img.npy"
                mask_path = ROOT_DIR / str(i) / "mask.npy"

                # Lire les fichiers sous forme d'octets bruts (évite le réencodage)
                with open(img_path, "rb") as f_img:
                    img_bytes = f_img.read()

                with open(mask_path, "rb") as f_mask:
                    mask_bytes = f_mask.read()

                sink.write({
                    "__key__": f"drand_{i}", # :06d
                    "img.npy": img_bytes,
                    "mask.npy": mask_bytes,
                })

    print("Conversion terminée !")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dreal", action='store_true')
    parser.add_argument("--drand", action='store_true')
    parser.add_argument("--dir", type=str, default="data")
    args = parser.parse_args()
    dir = Path(args.dir)
    assert dir.exists()

    if args.dreal:
        process_dreal()
    if args.drand:
        process_drand(dir)