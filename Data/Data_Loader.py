from monai.data import DataLoader, Dataset
import torch
import numpy as np
from utils import set_seed
import os
import random
from monai.transforms import LoadImage
from PIL import Image
from Data.transforms import train_transforms_1_channel, test_transforms_1_channel, train_transforms_multi_channel, test_transforms_multi_channel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import pandas as pd
from utils import merge_image_and_feature_data

IMAGE_EXTENSIONS = {'.bmp', '.png', '.jpg', '.jpeg', '.tiff', '.tif'}

set_seed(42)


def get_dataset_3ch(data_dir, Needed_modalities=()):
    train_data, test_data = [], []

    for mode in ('train', 'test'):
        mode_data = train_data if mode == 'train' else test_data
        base_dir = os.path.join(data_dir, mode)

        ref_mod = Needed_modalities[0]
        ref_dir = os.path.join(base_dir, ref_mod)

        for class_name in os.listdir(ref_dir):
            label = int(class_name)
            if label == 2:
                label = 1

            dirs = {
                'cc': os.path.join(base_dir, 'cc', class_name),
                'sup': os.path.join(base_dir, 'sup', class_name),
                'deep': os.path.join(base_dir, 'deep', class_name),
                'perf_maps': os.path.join(base_dir, 'perf_maps', class_name),
                'Vessels': os.path.join(base_dir, 'Vessels', class_name),
                'Capillaries': os.path.join(base_dir, 'Capillaries', class_name)
            }

            files = {
                k: {f.split('_')[0]: f for f in os.listdir(dirs[k])}
                for k in Needed_modalities
            }

            for patient_id in files[ref_mod]:
                patient_paths = {}
                valid = True

                for m in Needed_modalities:
                    f = files[m].get(patient_id)
                    if f is None:
                        valid = False
                        break
                    patient_paths[m] = os.path.join(dirs[m], f)

                if valid:
                    mode_data.append({
                        'image_paths': patient_paths,
                        'label': label,
                        'ID' : patient_id
                    })

    return train_data, test_data




def get_dataloader(data_dir, batch_size, num_workers, Needed_modalities=()):
    train_data, test_data = get_dataset_3ch(data_dir, Needed_modalities)

    train_ds = Dataset(data=train_data, transform=train_transforms_1_channel if len(Needed_modalities) == 1 else train_transforms_multi_channel)
    test_ds  = Dataset(data=test_data, transform=test_transforms_1_channel if len(Needed_modalities) == 1 else test_transforms_multi_channel)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def get_k_fold_data_loaders(data_dir, k, batch_size, num_workers, Needed_modalities=()):
    train_data, test_data = get_dataset_3ch(data_dir, Needed_modalities)
    data= train_data + test_data
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_loaders = []

    for train_index, test_index in kf.split(data):
        train_data = [data[i] for i in train_index]
        test_data  = [data[i] for i in test_index]

        train_ds = Dataset(data=train_data, transform=train_transforms_1_channel if len(Needed_modalities) == 1 else train_transforms_multi_channel)
        test_ds  = Dataset(data=test_data, transform=test_transforms_1_channel if len(Needed_modalities) == 1 else test_transforms_multi_channel)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        fold_loaders.append((train_loader, test_loader))

    return fold_loaders


# ---------------------------------------------------------------------------
# Inference data loaders (no train/test split required)
# ---------------------------------------------------------------------------

def get_inference_dataset(data_dir, Needed_modalities=()):
    """
    Load unlabeled images for pure prediction.

    Expected folder layout:
        data_dir/
        ├── sup/
        │   ├── PAT001_sup.bmp
        │   └── PAT002_sup.bmp
        ├── deep/
        │   ├── PAT001_deep.bmp
        │   └── PAT002_deep.bmp
        └── cc/   (and any other requested modalities)

    Patient ID is extracted as the part of the filename before the first '_'.
    IDs must match across all modality folders.
    """
    ref_mod = Needed_modalities[0]
    ref_dir = os.path.join(data_dir, ref_mod)

    if not os.path.isdir(ref_dir):
        raise FileNotFoundError(f"Modality folder not found: {ref_dir}")

    files = {}
    for m in Needed_modalities:
        mod_dir = os.path.join(data_dir, m)
        files[m] = {
            fname.split('_')[0]: fname
            for fname in os.listdir(mod_dir)
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS
        }

    samples = []
    for patient_id in files[ref_mod]:
        patient_paths = {}
        valid = True
        for m in Needed_modalities:
            f = files[m].get(patient_id)
            if f is None:
                valid = False
                break
            patient_paths[m] = os.path.join(data_dir, m, f)
        if valid:
            samples.append({'image_paths': patient_paths, 'label': 0, 'ID': patient_id})

    return samples


def get_inference_dataloader(data_dir, batch_size, num_workers, Needed_modalities=()):
    """DataLoader for unlabeled inference data (see get_inference_dataset for folder layout)."""
    samples = get_inference_dataset(data_dir, Needed_modalities)
    transform = test_transforms_1_channel if len(Needed_modalities) == 1 else test_transforms_multi_channel
    ds = Dataset(data=samples, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def get_labeled_inference_dataset(data_dir, Needed_modalities=(), remap_class2=True):
    """
    Load labeled images for inference + metric evaluation.

    Expected folder layout:
        data_dir/
        ├── sup/
        │   ├── 0/    <- class 0 images
        │   ├── 1/    <- class 1 images
        │   └── 2/    <- class 2 images
        ├── deep/
        │   ├── 0/
        │   ├── 1/
        │   └── 2/
        └── cc/  (and any other requested modalities)

    Patient ID is extracted as the part of the filename before the first '_'.
    IDs must match across all modality folders and within the same class subfolder.

    remap_class2: if True, label 2 is remapped to 1 (use for 2-class models).
                  if False, label 2 is kept as-is (use for 3-class models).
    """
    ref_mod = Needed_modalities[0]
    ref_dir = os.path.join(data_dir, ref_mod)

    if not os.path.isdir(ref_dir):
        raise FileNotFoundError(f"Modality folder not found: {ref_dir}")

    samples = []
    for class_name in os.listdir(ref_dir):
        class_path = os.path.join(ref_dir, class_name)
        if not os.path.isdir(class_path) or not class_name.isdigit():
            continue

        label = int(class_name)
        if remap_class2 and label == 2:
            label = 1

        dirs = {m: os.path.join(data_dir, m, class_name) for m in Needed_modalities}
        if not all(os.path.isdir(d) for d in dirs.values()):
            continue

        files = {
            k: {
                fname.split('_')[0]: fname
                for fname in os.listdir(dirs[k])
                if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS
            }
            for k in Needed_modalities
        }

        for patient_id in files[ref_mod]:
            patient_paths = {}
            valid = True
            for m in Needed_modalities:
                f = files[m].get(patient_id)
                if f is None:
                    valid = False
                    break
                patient_paths[m] = os.path.join(dirs[m], f)
            if valid:
                samples.append({'image_paths': patient_paths, 'label': label, 'ID': patient_id})

    return samples


def get_labeled_inference_dataloader(data_dir, batch_size, num_workers, Needed_modalities=(), remap_class2=True):
    """DataLoader for labeled inference data (see get_labeled_inference_dataset for folder layout)."""
    samples = get_labeled_inference_dataset(data_dir, Needed_modalities, remap_class2=remap_class2)
    transform = test_transforms_1_channel if len(Needed_modalities) == 1 else test_transforms_multi_channel
    ds = Dataset(data=samples, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)