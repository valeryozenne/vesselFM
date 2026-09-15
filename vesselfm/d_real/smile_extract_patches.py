from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np


def load_subject_volume(subject_dir: Path) -> tuple[np.ndarray, dict]:
	slice_paths = sorted(subject_dir.glob("*.npy"))
	if not slice_paths:
		raise FileNotFoundError(f"No .npy slices found in {subject_dir}")

	slices = [np.load(slice_path) for slice_path in slice_paths if slice_path.name != "metadata.json"]
	volume = np.stack(slices, axis=0)

	metadata_path = subject_dir / "metadata.json"
	metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
	return volume, metadata


def metadata_to_affine(metadata: dict) -> np.ndarray:
	spacing = metadata.get("spacing", [1.0, 1.0, 1.0])
	origin = metadata.get("origin", [0.0, 0.0, 0.0])
	direction = metadata.get("direction", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

	direction_matrix = np.asarray(direction, dtype=np.float64).reshape(3, 3)
	affine = np.eye(4, dtype=np.float64)
	affine[:3, :3] = direction_matrix @ np.diag(np.asarray(spacing, dtype=np.float64))
	affine[:3, 3] = np.asarray(origin, dtype=np.float64)
	return affine


def save_nifti(volume: np.ndarray, metadata: dict, output_path: Path, is_mask: bool) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)

	affine = metadata_to_affine(metadata)
	data = volume.astype(np.uint8 if is_mask else np.int16, copy=False)
	image = nib.Nifti1Image(data, affine=affine)
	nib.save(image, output_path)


def split_subjects(subjects: list[Path], train_count: int, val_count: int) -> dict[str, list[Path]]:
	if train_count + val_count > len(subjects):
		raise ValueError(
			f"Requested {train_count + val_count} train/val subjects but only {len(subjects)} are available"
		)

	train_subjects = subjects[:train_count]
	val_subjects = subjects[train_count : train_count + val_count]
	test_subjects = subjects[train_count + val_count :]
	return {"train": train_subjects, "val": val_subjects, "test": test_subjects}


def export_smile_dataset(input_root: Path, output_root: Path, train_count: int = 3, val_count: int = 1) -> None:
	image_root = input_root / "imagesTr"
	label_root = input_root / "labelsTr"

	if not image_root.exists():
		raise FileNotFoundError(f"Missing SMILE image folder: {image_root}")
	if not label_root.exists():
		raise FileNotFoundError(f"Missing SMILE label folder: {label_root}")

	subjects = sorted([path for path in image_root.iterdir() if path.is_dir()])
	if not subjects:
		raise FileNotFoundError(f"No subject folders found under {image_root}")

	splits = split_subjects(subjects, train_count=train_count, val_count=val_count)
	print(f"Found {len(subjects)} SMILE subjects: {[subject.name for subject in subjects]}")
	print({split_name: [subject.name for subject in subjects_in_split] for split_name, subjects_in_split in splits.items()})

	for split_name, subjects_in_split in splits.items():
		for subject_index, image_subject_dir in enumerate(subjects_in_split):
			label_subject_dir = label_root / image_subject_dir.name
			if not label_subject_dir.exists():
				raise FileNotFoundError(f"Missing label folder for {image_subject_dir.name}: {label_subject_dir}")

			image_volume, image_metadata = load_subject_volume(image_subject_dir)
			label_volume, label_metadata = load_subject_volume(label_subject_dir)

			if image_volume.shape != label_volume.shape:
				raise ValueError(
					f"Shape mismatch for {image_subject_dir.name}: image {image_volume.shape} vs label {label_volume.shape}"
				)

			output_subject_dir = output_root / split_name / f"{subject_index:01d}"
			save_nifti(image_volume, image_metadata, output_subject_dir / "img.nii.gz", is_mask=False)
			save_nifti(label_volume > 0, label_metadata, output_subject_dir / "mask.nii.gz", is_mask=True)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Export SMILE converted slices to finetune-ready NIfTI volumes.")
	parser.add_argument(
		"--input-root",
		type=Path,
		default=Path("data/converted_real/SMILE"),
		help="Path to the converted SMILE dataset root.",
	)
	parser.add_argument(
		"--output-root",
		type=Path,
		default=Path("data/finetune/smile"),
		help="Path where train/val/test NIfTI volumes will be written.",
	)
	parser.add_argument(
		"--train-count",
		type=int,
		default=3,
		help="Number of subjects to place in the train split.",
	)
	parser.add_argument(
		"--val-count",
		type=int,
		default=1,
		help="Number of subjects to place in the validation split.",
	)
	parser.add_argument(
		"--clean-output",
		action="store_true",
		help="Remove the output root before exporting the dataset.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.clean_output and args.output_root.exists():
		shutil.rmtree(args.output_root)

	export_smile_dataset(
		input_root=args.input_root,
		output_root=args.output_root,
		train_count=args.train_count,
		val_count=args.val_count,
	)


if __name__ == "__main__":
	main()
