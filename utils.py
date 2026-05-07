import random
import numpy as np  
import torch
import monai
import gc
import os
from collections import OrderedDict
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score as kappa_score)
from scipy.special import softmax
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    monai.utils.set_determinism(seed=seed)

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect() 

def specificity_score(cm):
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)

def evaluate(preds, labels, epoch="NA", mode="val", save_predictions=False, save_path=None):
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)

    if torch.is_tensor(preds):
        logits = preds
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        pred_labels = torch.argmax(logits, dim=1).detach().cpu().numpy()
    else:
        logits = np.asarray(preds)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        pred_labels = np.argmax(logits, axis=1)
    num_classes = probs.shape[1]
    acc = accuracy_score(labels_np, pred_labels)
    f1 = f1_score(labels_np, pred_labels, average="weighted", zero_division=0.0)
    mcc = matthews_corrcoef(labels_np, pred_labels)
    kappa = kappa_score(labels_np, pred_labels)
    if num_classes == 2:
        auroc = roc_auc_score(labels_np, probs[:, 1])
        ap = average_precision_score(labels_np, probs[:, 1])
    else:
        true_onehot = label_binarize(labels_np, classes=np.arange(num_classes))
        auroc = roc_auc_score(true_onehot, probs, multi_class="ovr", average="weighted")
        ap = average_precision_score(true_onehot, probs, average="weighted")

    avg_loss = 0

    if save_predictions:
        raise NotImplementedError("Patient-level aggregation is not implemented in this evaluate().")

    return OrderedDict({
        "epoch": epoch,
        "loss": avg_loss,
        "acc": acc,
        "auroc": auroc,
        "ap": ap,
        "f1": f1,
        "mcc": mcc,
        'kappa': kappa
    }
    )


def plot_confusion_matrix(all_fold_dir):
    all_cms = []
    for i,fold_dir in enumerate(os.listdir(all_fold_dir)):
        fold_path= os.path.join(all_fold_dir, fold_dir)
        fold_res = torch.load(fold_path,weights_only=False)
        best_epoch = fold_res['stats']["best_epoch"]
        cm = fold_res['stats']["val"]["confusion_matrix"][best_epoch]
        all_cms.append(cm)
    plt.figure(figsize=(16, 9))
    for i,cm in enumerate(all_cms):
        plt.subplot(2, int(len(all_cms)/2), i+1)
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
        plt.title(f'Mean Confusion Matrix fold{i+1}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
    plt.show()

def merge_image_and_feature_data(image_data, feature_data):
    feature_dict = {sample["ID"]: sample for sample in feature_data}

    merged = []
    for img_sample in image_data:
        ID = img_sample["ID"]

        if ID not in feature_dict:
            continue  # skip if mismatch

        feat_sample = feature_dict[ID]

        merged.append({
            "ID": ID,
            "image": img_sample["image"],           # list of image paths
            "features": feat_sample["features"],    # list of .npy feature files
            "label": img_sample["label"]
        })
    return merged
