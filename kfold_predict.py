import os
import torch 
import numpy as np
from Training.Models import Custom_Net
from Training.Engine import Engine
from Data.Data_Loader import get_dataloader
from utils import set_seed,evaluate

data_loader = get_dataloader(
    data_dir= r'C:\Users\me510671\Desktop\Work\Datasets\RASTA\RASTA2',
    batch_size= 4,
    num_workers= 4,
    Needed_modalities=('sup','deep','cc','perf_map')
)
test_loader = data_loader[1]

model = Custom_Net(num_classes=3, input_shape=[4,256,256], pretrained=False, dropout=0.0, Adaptive_input=True, version='b0')
kfold_model_path = r'C:\Users\me510671\Desktop\Work\Codes\RASTA-img-classification-Nov25\img-classification-Nov25\results\New\Custom_Net_lr0.0001_5_fold_4ch'
engine = Engine(model=model, optimizer=None, loss_fn=None, data_loader=data_loader)

def kfold_voting(model, folds_path, agg_method='mean', engine=engine, test_loader=test_loader):
    fold_results = []   # ← OUTSIDE the loop
    fold_labels = None  # ← OUTSIDE, use None as "not yet set" flag

    for fold in os.listdir(folds_path):
        fold_path = os.path.join(folds_path, fold)
        if os.path.isdir(fold_path):
            model_path = os.path.join(fold_path, 'final_model.pth')
            if os.path.isfile(model_path):
                print(f"Loading model from {model_path}...")
                checkpoint = torch.load(model_path, map_location='cpu')
                state_dict = checkpoint['network'] if 'network' in checkpoint else checkpoint
                model.load_state_dict(state_dict)

                test_m, predictions, labels = engine.test(data_loader=test_loader, mode="val")
                fold_results.append(torch.stack(predictions))  # [n_samples, n_classes]

                if fold_labels is None:  # ← store once, labels are same across folds
                    fold_labels = torch.stack(labels)  # [n_samples]

                print(
                    f"Val | acc: {test_m['acc']:.4f} AUROC: {test_m['auroc']:.4f} AP: {test_m['ap']:.4f} "
                    f"Kappa: {test_m['kappa']:.4f} F1: {test_m['f1']:.4f} MCC: {test_m['mcc']:.4f}"
                )
            else:
                print(f"No model found at {model_path}, skipping fold.")
        else:
            print(f"{fold_path} is not a directory, skipping.")

    if not fold_results:
        raise RuntimeError("No fold results collected. Check your folds path.")

    if agg_method == 'mean':
        aggregated_logits = torch.mean(torch.stack(fold_results), dim=0)  # [n_samples, n_classes]
        aggregated_predictions = torch.argmax(aggregated_logits, dim=1)   # [n_samples]
        eval_input = aggregated_logits  # ← pass logits to evaluate

    elif agg_method == 'majority':
        fold_classes = torch.stack([torch.argmax(f, dim=1) for f in fold_results])  # [n_folds, n_samples]
        aggregated_predictions = torch.mode(fold_classes, dim=0).values              # [n_samples]
        # majority has no logits — use mean logits as proxy for probability estimates
        eval_input = torch.mean(torch.stack(fold_results), dim=0)  # [n_samples, n_classes]

    else:
        raise ValueError(f"Unsupported aggregation method: {agg_method}")
    agg_results = evaluate(eval_input, fold_labels, mode="val")

    print(
        f"Aggregated Val | acc: {agg_results['acc']:.4f} AUROC: {agg_results['auroc']:.4f} "
        f"AP: {agg_results['ap']:.4f} Kappa: {agg_results['kappa']:.4f} "
        f"F1: {agg_results['f1']:.4f} MCC: {agg_results['mcc']:.4f}"
    )
    return aggregated_predictions, agg_results
            

if __name__ == "__main__":
    aggregated_predictions, agg_results = kfold_voting(model, kfold_model_path, agg_method='majority', engine=engine, test_loader=test_loader)
