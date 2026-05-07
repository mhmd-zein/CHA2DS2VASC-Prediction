#!/usr/bin/env python3
"""
RASTA Retinal Classification — Inference

Organize your images in the data/ folder as described in README.md, then run:
    python inference.py

Results are saved to predictions/predictions.csv.
"""

import os
import sys
import torch
import pandas as pd

from Training.Models import Custom_Net
from Data.Data_Loader import get_inference_dataloader, get_labeled_inference_dataloader
from utils import set_seed, evaluate

MODEL_PATH  = 'Results/model/final_model.pth'
DATA_DIR    = 'data'
MODALITIES  = ('sup', 'deep', 'cc')
NUM_CLASSES = 3
OUTPUT_DIR  = 'predictions'
BATCH_SIZE  = 4
SEED        = 42


def _has_labels(data_dir, modalities):
    """Auto-detect labeled layout: True if modality subfolders contain digit-named subdirs."""
    ref = os.path.join(data_dir, modalities[0])
    return os.path.isdir(ref) and any(
        os.path.isdir(os.path.join(ref, d)) and d.isdigit()
        for d in os.listdir(ref)
    )


def main():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load model ───────────────────────────────────────────────────────────
    if not os.path.isfile(MODEL_PATH):
        sys.exit(f"Model file not found at '{MODEL_PATH}'. Check that the repository was cloned correctly.")

    model = Custom_Net(
        num_classes=NUM_CLASSES,
        input_shape=[3, 256, 256],
        pretrained=False,
        dropout=0.0,
        Adaptive_input=False,
        version='b0',
    )
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state_dict = checkpoint['network'] if 'network' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded.")

    # ── Load data ────────────────────────────────────────────────────────────
    labeled = _has_labels(DATA_DIR, MODALITIES)
    if labeled:
        print("Ground truth labels detected — evaluation metrics will be computed.")
        loader = get_labeled_inference_dataloader(DATA_DIR, BATCH_SIZE, 0, MODALITIES)
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
    for c in range(NUM_CLASSES):
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
    for c in range(NUM_CLASSES):
        n = int((pred_cls == c).sum())
        print(f"  Class {c}: {n} patient(s) ({100 * n / len(pred_cls):.1f}%)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, 'predictions.csv')
    results.to_csv(out_path, index=False)
    print(f"\nPredictions saved to: {out_path}")


if __name__ == '__main__':
    main()
