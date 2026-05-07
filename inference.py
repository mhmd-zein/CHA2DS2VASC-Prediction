#!/usr/bin/env python3
"""
RASTA Retinal Classification — Inference

Organize your images in the data/ folder as described in README.md, then run:
    python inference.py --num_classes 2
    python inference.py --num_classes 3

Results are saved to predictions/predictions.csv.
"""

import argparse
import os
import sys
import torch
import pandas as pd

from Training.Models import Custom_Net
from Data.Data_Loader import get_inference_dataloader, get_labeled_inference_dataloader
from utils import set_seed, evaluate

# ── Model paths (fill in the folder names before deploying) ─────────────────
MODEL_PATH = {
    2: 'Results/ /final_model.pth',   # ← path to the 2-class model
    3: 'Results/ /final_model.pth',   # ← path to the 3-class model
}
# ────────────────────────────────────────────────────────────────────────────

DATA_DIR   = 'data'
MODALITIES = ('sup', 'deep', 'cc')
OUTPUT_DIR = 'predictions'
BATCH_SIZE = 4
SEED       = 42


def parse_args():
    parser = argparse.ArgumentParser(description='RASTA Retinal Classification — Inference')
    parser.add_argument(
        '--num_classes', type=int, required=True, choices=[2, 3],
        help='Number of classes: 2 (binary) or 3 (multi-class).'
    )
    return parser.parse_args()


def _has_labels(data_dir, modalities):
    """Auto-detect labeled layout: True if modality subfolders contain digit-named subdirs."""
    ref = os.path.join(data_dir, modalities[0])
    return os.path.isdir(ref) and any(
        os.path.isdir(os.path.join(ref, d)) and d.isdigit()
        for d in os.listdir(ref)
    )


def main():
    args = parse_args()
    num_classes = args.num_classes
    model_path  = MODEL_PATH[num_classes]

    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device:      {device}")
    print(f"Mode:        {num_classes}-class")

    # ── Load model ───────────────────────────────────────────────────────────
    if not os.path.isfile(model_path):
        sys.exit(f"Model file not found at '{model_path}'. Check that the repository was cloned correctly.")

    model = Custom_Net(
        num_classes=num_classes,
        input_shape=[3, 256, 256],
        pretrained=False,
        dropout=0.0,
        Adaptive_input=False,
        version='b0',
    )
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['network'] if 'network' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded from: {model_path}")

    # ── Load data ────────────────────────────────────────────────────────────
    labeled     = _has_labels(DATA_DIR, MODALITIES)
    remap_cls2  = (num_classes == 2)   # remap class 2 → 1 only for the binary model

    if labeled:
        print("Ground truth labels detected — evaluation metrics will be computed.")
        loader = get_labeled_inference_dataloader(
            DATA_DIR, BATCH_SIZE, 0, MODALITIES, remap_class2=remap_cls2
        )
    else:
        loader = get_inference_dataloader(DATA_DIR, BATCH_SIZE, 0, MODALITIES)

    if len(loader.dataset) == 0:
        sys.exit(
            f"No samples found in '{DATA_DIR}/'. "
            "Check that your images are organized as described in README.md."
        )
    print(f"Found {len(loader.dataset)} patient(s). Running inference...")

    # ── Inference ────────────────────────────────────────────────────────────
    all_preds, all_labels, all_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            outputs = model(images)
            all_preds.extend(outputs.detach().cpu())
            all_ids.extend(batch['ID'])
            if labeled:
                all_labels.extend(batch['label'].cpu())

    pred_tensor = torch.stack(all_preds)
    probs       = torch.softmax(pred_tensor, dim=1).numpy()
    pred_cls    = torch.argmax(pred_tensor, dim=1).numpy()

    # ── Results table ────────────────────────────────────────────────────────
    results = pd.DataFrame({'patient_id': all_ids, 'predicted_class': pred_cls})
    for c in range(num_classes):
        results[f'prob_class_{c}'] = probs[:, c]

    if labeled:
        labels_tensor = torch.stack(all_labels)
        results['true_label'] = labels_tensor.numpy()
        m = evaluate(pred_tensor, labels_tensor, mode='test')
        print("\n=== Evaluation Metrics ===")
        print(f"  Accuracy:        {m['acc']:.4f}")
        print(f"  AUROC:           {m['auroc']:.4f}")
        print(f"  Avg Precision:   {m['ap']:.4f}")
        print(f"  F1 (weighted):   {m['f1']:.4f}")
        print(f"  MCC:             {m['mcc']:.4f}")
        print(f"  Cohen's Kappa:   {m['kappa']:.4f}")

    print("\n=== Prediction Summary ===")
    for c in range(num_classes):
        n = int((pred_cls == c).sum())
        print(f"  Class {c}: {n} patient(s) ({100 * n / len(pred_cls):.1f}%)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, 'predictions.csv')
    results.to_csv(out_path, index=False)
    print(f"\nPredictions saved to: {out_path}")


if __name__ == '__main__':
    main()
