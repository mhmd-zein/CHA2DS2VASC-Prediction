import os
import json
import numpy as np

exp_path = r'Results\Custom_Net_lr0.0001_5_fold_sup_deep_cc'

def get_val(stats, key):
    v = stats['val'][key]
    return v[-1] if isinstance(v, list) else v
exp_accs=[]
exp_aurocs=[]
exp_aps=[]
exp_kappas=[]
exp_f1s=[]
exp_mccs=[]
for fold in os.listdir(exp_path):
    fold_path = os.path.join(exp_path, fold)
    if os.path.isdir(fold_path):
        result_path = os.path.join(fold_path, 'stats.json')
        if os.path.isfile(result_path):
            with open(result_path, 'r') as f:
                stats = json.load(f)
                fold_acc = get_val(stats,'acc')
                fold_auroc = get_val(stats,'auroc')
                fold_ap = get_val(stats,'ap')
                fold_kappa = get_val(stats,'kappa')
                fold_f1 = get_val(stats,'f1')
                fold_mcc = get_val(stats,'mcc')
                exp_accs.append(fold_acc)
                exp_aurocs.append(fold_auroc)
                exp_aps.append(fold_ap)
                exp_kappas.append(fold_kappa)
                exp_f1s.append(fold_f1)
                exp_mccs.append(fold_mcc)
                
                print(f"Fold {fold} | Val | acc: {fold_acc:.4f} AUROC: {fold_auroc:.4f} AP: {fold_ap:.4f} Kappa: {fold_kappa:.4f} F1: {fold_f1:.4f} MCC: {fold_mcc:.4f}")

a,r,p,k,f,m = [np.array(x) for x in [exp_accs,exp_aurocs,exp_aps,exp_kappas,exp_f1s,exp_mccs]]
print(
    f"Experiment {exp_path} | Mean Val | "
    f"acc: {a.mean():.4f}±{a.std():.4f}  AUROC: {r.mean():.4f}±{r.std():.4f}  "
    f"AP: {p.mean():.4f}±{p.std():.4f}  Kappa: {k.mean():.4f}±{k.std():.4f}  "
    f"F1: {f.mean():.4f}±{f.std():.4f}  MCC: {m.mean():.4f}±{m.std():.4f}"
)