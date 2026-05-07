from pyexpat import model
from sched import scheduler
import torch
import numpy as np
from utils import set_seed, clear_memory, evaluate
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
# import wandb
from pathlib import Path
import numpy as np
import json
import os
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class Engine:
    def __init__(self, model, optimizer, loss_fn, data_loader ,scheduler=None, scheduler_step='epoch', K_fold=False, agg_method='mean', exp_name=None, run_explainability=False, XAI_target_class=None):
        self.network = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.network.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler=scheduler
        self.scheduler_step = scheduler_step
        self.agg_method = agg_method
        self.exp_title = exp_name
        self.run_explainability = run_explainability
        self.target_class = XAI_target_class
        self.K_fold = K_fold
        if K_fold:
            self.fold_data_loader = data_loader

        else:
            self.train_loader=data_loader[0]       
            self.val_loader=data_loader[1]
        # wandb.init(
        #     project="RASTA_Classification",
        #     name=(
        #         f"{type(self.network).__name__}_"
        #         f"lr{self.optimizer.param_groups[0]['lr']}_"
        #         f"bs{self.train_loader.batch_size}"
        #         f'_AggMethod_{self.agg_method}'
        #     ),
        #     config={
        #         "batch_size": self.train_loader.batch_size,
        #         "optimizer": type(self.optimizer).__name__,
        #         "scheduler": type(self.scheduler).__name__ if self.scheduler else None,
        #         "loss_fn": type(self.loss_fn).__name__,
        #     }
        # )
        # Fix 1: correct the f-string so the condition applies to the whole expression
        self.exp_name = (
            f"{type(self.network).__name__}_"
            # f"lr{self.optimizer.param_groups[0]['lr']}"
            + (f"_{self.exp_title}" if self.exp_title else "")
        )
        # wandb.watch(self.network, log="all", log_freq=100)
    def _resolve_save_path(self, save_path):
        save_path = Path(save_path)
        exp_dir = save_path/ self.exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir   

    def run_k_fold(self, epochs, save_path, verbose=True):
        print(f"Experiment name: {self.exp_name}")
        all_fold_results = []
        for fold_idx, (train_loader, val_loader) in enumerate(self.fold_data_loader):
            print(f"\n===== Fold {fold_idx + 1} / {len(self.fold_data_loader)} =====")
            model_fold = deepcopy(self.network)
            optimizer_fold = type(self.optimizer)(model_fold.parameters(), **self.optimizer.defaults)
            scheduler_fold = None
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    scheduler_fold = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer_fold,
                        T_0=5*len(train_loader)
                    )
                else:
                    try:
                        scheduler_fold = type(self.scheduler)(optimizer_fold, **self.scheduler.defaults)
                    except AttributeError:
                        scheduler_fold = None

            # Create a fold-specific Engine
            fold_engine = Engine(
                model=model_fold,
                optimizer=optimizer_fold,
                loss_fn=self.loss_fn,
                data_loader=(train_loader, val_loader),
                scheduler=scheduler_fold,
                scheduler_step=self.scheduler_step,
                K_fold=False,
                exp_name= self.exp_title,
                run_explainability=self.run_explainability,
                XAI_target_class=self.target_class,
            )

            fold_results = fold_engine.train(
                epochs=epochs,
                save_path= save_path,
                verbose=verbose,
                fold = fold_idx + 1
            )
            all_fold_results.append(fold_results)

        return all_fold_results
    
    def train(self, epochs, save_path, results=None, verbose=True , fold=None):
        if not self.K_fold:
            print(f"Experiment name: {self.exp_name}")
        results = results or {
            "best_epoch": -1,
            "train": {"loss": [], "acc": [], "auroc": [], "ap": [], "f1": [], "mcc": [],'kappa': []},
            "val": {"acc": [], "auroc": [], "ap": [], "f1": [], "mcc": [],'kappa': []},
        }
        best_acc = results["val"]["acc"][results["best_epoch"]] if results["best_epoch"] >= 0 else 0.0

        for epoch in range(epochs):
            self.network.train()
            all_labels = []
            all_preds = []
            train_loss = 0.0

            for j, batch in enumerate(tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training")):
            # for batch in self.train_loader:
                clear_memory()
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.network(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * images.size(0)
                all_preds.extend(outputs.detach().cpu())
                all_labels.extend(labels.detach().cpu())
                if self.scheduler is not None and self.scheduler_step=='batch':
                    self.scheduler.step()
            avg_train_loss = train_loss / len(self.train_loader.dataset)
            train_m = evaluate(torch.stack(all_preds), torch.stack(all_labels), epoch=epoch, mode="train")
            val_m, val_cams, val_ids = self.test(data_loader=self.val_loader, agg_method=self.agg_method, epoch=epoch, mode="val", run_explainability=self.run_explainability, xai_class=self.target_class)

            if self.scheduler is not None and self.scheduler_step=='epoch':
                self.scheduler.step()

            results["train"]["loss"].append(avg_train_loss)
            results["train"]["acc"].append(train_m["acc"])
            results["train"]["auroc"].append(train_m["auroc"])
            results["train"]["ap"].append(train_m["ap"])
            results["train"]["f1"].append(train_m["f1"])
            results["train"]["mcc"].append(train_m["mcc"])
            results["train"]['kappa'].append(train_m['kappa'])

            results["val"]["acc"].append(val_m["acc"])
            results["val"]["auroc"].append(val_m["auroc"])
            results["val"]["ap"].append(val_m["ap"])
            results["val"]["f1"].append(val_m["f1"])
            results["val"]["mcc"].append(val_m["mcc"])
            results["val"]['kappa'].append(val_m['kappa'])

            # wandb.log({
            # "Train Loss": avg_train_loss,
            # "Train acc": train_m["acc"],
            # "Train AUROC": train_m["auroc"],
            # "Train AP": train_m["ap"],
            # "Train F1": train_m["f1"],
            # "Train MCC": train_m["mcc"],
            # "Val acc": val_m["acc"],
            # "Val AUROC": val_m["auroc"],
            # "Val AP": val_m["ap"],
            # "Val F1": val_m["f1"],
            # "Val MCC": val_m["mcc"],
            # })
            # wandb.log({
            #     "train/confusion_matrix": wandb.plot.confusion_matrix(
            #         preds=torch.argmax(torch.stack(all_preds), dim=1).numpy(),
            #         y_true=torch.stack(all_labels).numpy(),
            #         class_names=["class_0", "class_1"]
            #     ),
            # })
            

            print(
                f"Train | Loss: {avg_train_loss:.4f} | "
                f"acc: {train_m['acc']:.4f} AUROC: {train_m['auroc']:.4f} AP: {train_m['ap']:.4f}  Kappa: {train_m['kappa']:.4f} "
                f"F1: {train_m['f1']:.4f} MCC: {train_m['mcc']:.4f}"
            )
            print(
                f"Val | acc: {val_m['acc']:.4f} AUROC: {val_m['auroc']:.4f} AP: {val_m['ap']:.4f}  Kappa: {val_m['kappa']:.4f} "
                f"F1: {val_m['f1']:.4f} MCC: {val_m['mcc']:.4f}"
            )
            if verbose:
                plt.figure(figsize=(12,4))
                plt.subplot(1,3,1)
                plt.plot(results["train"]["acc"], label="Train acc")
                plt.plot(results["val"]["acc"], label="Validation acc")
                plt.xlabel('Epoch')
                plt.ylabel('balanced_acc')
                plt.title('Training and Validation balanced_acc')
                plt.legend()


            if val_m["acc"] > best_acc:
                results["best_epoch"] = epoch
                best_acc = val_m["acc"]
                self.save(save_path, trainable=True, stats=results, fold=fold)
                # wandb.save(str(save_path))  # saves the model to W&B cloud
                print(f"Model saved at epoch {epoch+1} with val balanced acc { val_m['acc']:.4f}")
                if self.run_explainability and val_cams:
                    self.save_heatmaps(val_cams, val_ids, path=save_path, fold=fold)
        return results
    

    def test(self, save_path=None, data_loader=None, agg_method='mean', 
            epoch="NA", mode="val", run_explainability=False, xai_class=None):
        if save_path:
            stats = self.load(save_path, return_stats=True)
            print(f"Loaded model from {save_path}, trained up to epoch {stats['best_epoch']+1}")
        
        self.network.eval()
        all_preds = []
        all_labels = []
        all_cams   = []  # optional, only populated if run_explainability=True
        all_ids= []
        # --- resolve target layer safely ---
        target_layer = [self.network.model.features[-1]]
        
        for batch in data_loader:
            clear_memory()
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # --- standard inference (no_grad) ---
            with torch.no_grad():
                outputs = self.network(images)
            all_preds.extend(outputs.detach().cpu())
            all_labels.extend(labels.cpu())
            all_ids.extend(batch['ID'])  # adjust to whatever key your dataset uses

            # --- XAI pass (separate, with gradients) ---
            if run_explainability:
                cam_maps = self._run_gradcam(
                    images=images,
                    target_layers=target_layer,
                    reshape_transform=None,
                    target_class=xai_class  # None = use predicted class per image
                )
                all_cams.extend(cam_maps)  # list of (H,W) numpy arrays

        m = evaluate(torch.stack(all_preds), torch.stack(all_labels), epoch=epoch, mode=mode)

        if run_explainability:
            return m, all_cams ,all_ids
        return m , None , all_ids


    def _run_gradcam(self, images, target_layers, reshape_transform, target_class=None):
        """
        Runs GradCAM on a batch. Gradients are enabled here.
        Returns a list of (H, W) numpy heatmaps, one per image in batch.
        """
        # Build per-image targets
        if target_class is not None:
            targets = [ClassifierOutputTarget(target_class)] * images.shape[0]
        else:
            # Use each image's own predicted class
            with torch.no_grad():
                preds = self.network(images).argmax(dim=1).cpu().tolist()
            targets = [ClassifierOutputTarget(c) for c in preds]

        with GradCAM(
            model=self.network,
            target_layers=target_layers,
            reshape_transform=reshape_transform
        ) as cam:
            grayscale_cams = cam(input_tensor=images, targets=targets)
            # grayscale_cams: (B, H, W) numpy array

        return [grayscale_cams[i] for i in range(len(grayscale_cams))]
    
    def save(self, path, trainable=True, stats=None, fold = None): 
        parent_path = self._resolve_save_path(path)
        if fold is not None:
            fold_path= os.path.join(parent_path, f'fold_{fold}')
            os.makedirs(fold_path, exist_ok=True)

        if fold is None:
            save_path = parent_path / "final_model.pth"
        else:
            save_path = Path(fold_path) / "final_model.pth"
        
        save_dict = { 'network': self.network.state_dict() } 

        if stats is not None:
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            stats_path = save_dir / "stats.json"
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=4)     

        if trainable:
            save_dict['optimizer'] = self.optimizer.state_dict()
            save_dict['scheduler'] = self.scheduler.state_dict() if self.scheduler else None

        torch.save(save_dict, save_path)


    def save_heatmaps(self, heatmaps, image_ids, path, fold=None):
        parent_path = self._resolve_save_path(path)
        save_dir = Path(os.path.join(parent_path, f'fold_{fold}')) if fold is not None else parent_path
        save_dir.mkdir(parents=True, exist_ok=True)

        # fixed filename — overwrites previous best by design
        heatmap_path = save_dir / "heatmaps.npz"
        np.savez(
            heatmap_path,
            heatmaps=np.stack(heatmaps),
            image_ids=np.array(image_ids)
        )
        print(f"Saved {len(heatmaps)} heatmaps to {heatmap_path}")

    def load(self, path, return_stats=True, fold=None):
        parent_path = self._resolve_save_path(path)
        if fold is not None:
            load_dir = Path(os.path.join(parent_path, f'fold_{fold}'))
        else:
            load_dir = parent_path

        # FIX 4: load the .pth file, not the directory
        checkpoint = torch.load(load_dir / "final_model.pth", map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint and checkpoint['scheduler'] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        if return_stats:
            # FIX 5: use json.load not torch.load for .json files
            stats_path = load_dir / "stats.json"
            if stats_path.exists():
                with open(stats_path, "r") as f:
                    return json.load(f)
            return None


