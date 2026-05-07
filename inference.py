#!/usr/bin/env python3
"""
RASTA Retinal Classification - Inference Script

Runs retinal disease classification on OCT/OCTA images using a pre-trained model.
Supports both prediction-only mode (no ground truth) and evaluation mode (with labels).

Usage: see README.md or run:  python inference.py --help
"""

import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd

from Training.Models import Custom_Net
from utils import set_seed, evaluate

SUPPORTED_MODALITIES = ('sup', 'deep', 'cc', 'perf_maps', 'Vessels', 'Capillaries')


def parse_args():
    parser = argparse.ArgumentParser(
        description='RASTA Retinal Classification - Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Single modality, prediction only (no ground truth labels):
  python inference.py \\
      --model_path Results/my_experiment/final_model.pth \\
      --data_dir   path/to/images \\
      --modalities sup \\
      --input_channels 1

  # Three-modality model, prediction only:
  python inference.py \\
      --model_path Results/my_experiment/final_model.pth \\
      --data_dir   path/to/images \\
      --modalities sup deep cc \\
      --input_channels 3

  # With ground truth labels (prints accuracy, AUROC, F1, ...):
  python inference.py \\
      --model_path     Results/my_experiment/final_model.pth \\
      --data_dir       path/to/images \\
      --modalities     sup deep cc \\
      --input_channels 3 \\
      --has_labels

  # Save results to a custom folder:
  python inference.py ... --output_dir /path/to/output
"""
    )

    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to the trained model checkpoint (.pth file).'
    )
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help=(
            'Root directory containing modality subfolders. '
            'Layout depends on --has_labels flag — see README.md for details.'
        )
    )
    parser.add_argument(
        '--modalities', nargs='+', required=True,
        choices=SUPPORTED_MODALITIES,
        help=(
            f'List of imaging modalities to use. Must match what was used during training. '
            f'Supported: {SUPPORTED_MODALITIES}'
        )
    )
    parser.add_argument(
        '--input_channels', type=int, required=True,
        help=(
            'Number of input channels (must match training). '
            'Typically: 1 for a single modality replicated to 3 channels, '
            'or N = number of modalities for multi-channel input.'
        )
    )
    parser.add_argument(
        '--num_classes', type=int, default=3,
        help='Number of output classes (default: 3, must match training).'
    )
    parser.add_argument(
        '--model_version', type=str, default='b0', choices=['b0', 'b4', 'v2s'],
        help='EfficientNet backbone version used during training (default: b0).'
    )
    parser.add_argument(
        '--dropout', type=float, default=0.0,
        help='Dropout rate used during training (default: 0.0).'
    )
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help='Batch size for inference (default: 4).'
    )
    parser.add_argument(
        '--num_workers', type=int, default=0,
        help=(
            'DataLoader worker processes (default: 0). '
            'Keep at 0 on Windows; can increase (e.g. 4) on Linux/macOS.'
        )
    )
    parser.add_argument(
        '--has_labels', action='store_true',
        help=(
            'Set this flag when data folders contain class subfolders (0/, 1/, 2/). '
            'Enables computation of accuracy, AUROC, F1, MCC, and Cohen\'s Kappa.'
        )
    )
    parser.add_argument(
        '--output_dir', type=str, default='predictions',
        help='Directory where predictions.csv will be saved (default: ./predictions).'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device: "cuda", "cpu", or None for auto-detect (default: auto).'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42).'
    )

    return parser.parse_args()


def load_model(model_path, num_classes, input_channels, model_version, dropout, device):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    adaptive_input = input_channels > 3
    model = Custom_Net(
        num_classes=num_classes,
        input_shape=[input_channels, 256, 256],
        pretrained=False,
        dropout=dropout,
        Adaptive_input=adaptive_input,
        version=model_version,
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['network'] if 'network' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_inference(model, data_loader, device, has_labels):
    all_preds, all_labels, all_ids = [], [], []

    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            outputs = model(images)
            all_preds.extend(outputs.detach().cpu())
            all_ids.extend(batch['ID'])
            if has_labels:
                all_labels.extend(batch['label'].cpu())

    return all_preds, (all_labels if has_labels else None), all_ids


def main():
    args = parse_args()
    set_seed(args.seed)

    # ── Device ──────────────────────────────────────────────────────────────
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Model ────────────────────────────────────────────────────────────────
    print(f"Loading model from: {args.model_path}")
    model = load_model(
        args.model_path,
        args.num_classes,
        args.input_channels,
        args.model_version,
        args.dropout,
        device,
    )
    print("Model loaded successfully.")

    # ── Data ─────────────────────────────────────────────────────────────────
    modalities = tuple(args.modalities)
    if args.has_labels:
        from Data.Data_Loader import get_labeled_inference_dataloader
        loader = get_labeled_inference_dataloader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            Needed_modalities=modalities,
        )
    else:
        from Data.Data_Loader import get_inference_dataloader
        loader = get_inference_dataloader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            Needed_modalities=modalities,
        )

    n_samples = len(loader.dataset)
    if n_samples == 0:
        sys.exit(
            "No samples found. Verify --data_dir and --modalities match your folder structure. "
            "See README.md for the expected layout."
        )
    print(f"Found {n_samples} patient(s). Running inference...")

    # ── Inference ────────────────────────────────────────────────────────────
    all_preds, all_labels, all_ids = run_inference(model, loader, device, args.has_labels)

    pred_tensor = torch.stack(all_preds)
    probs = torch.softmax(pred_tensor, dim=1).numpy()
    pred_classes = torch.argmax(pred_tensor, dim=1).numpy()

    # ── Build results table ──────────────────────────────────────────────────
    results = pd.DataFrame({
        'patient_id': all_ids,
        'predicted_class': pred_classes,
    })
    for c in range(args.num_classes):
        results[f'prob_class_{c}'] = probs[:, c]

    # ── Metrics (only when ground truth is available) ─────────────────────────
    if args.has_labels:
        labels_tensor = torch.stack(all_labels)
        results['true_label'] = labels_tensor.numpy()

        metrics = evaluate(pred_tensor, labels_tensor, mode='test')
        print("\n=== Evaluation Metrics ===")
        print(f"  Accuracy:        {metrics['acc']:.4f}")
        print(f"  AUROC:           {metrics['auroc']:.4f}")
        print(f"  Avg Precision:   {metrics['ap']:.4f}")
        print(f"  F1 (weighted):   {metrics['f1']:.4f}")
        print(f"  MCC:             {metrics['mcc']:.4f}")
        print(f"  Cohen's Kappa:   {metrics['kappa']:.4f}")

    # ── Prediction summary ───────────────────────────────────────────────────
    print("\n=== Prediction Summary ===")
    for c in range(args.num_classes):
        n = int((pred_classes == c).sum())
        print(f"  Class {c}: {n} patient(s) ({100 * n / len(pred_classes):.1f}%)")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'predictions.csv')
    results.to_csv(out_path, index=False)
    print(f"\nPredictions saved to: {out_path}")


if __name__ == '__main__':
    main()
