import argparse
from pathlib import Path
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from Training.Engine import Engine
from utils import set_seed
from Training.Models import build_model
from Data.Data_Loader import  get_dataloader , get_k_fold_data_loaders
from monai.losses import FocalLoss

# STUFF TO IMPROVE

# MAKE THE ADAPTIVE INPUT AUTOMATIC(SHAPE OF INPUT AUTOMATIC)
# MAKE THE DATALOADERS MORE CLEAR IN TERMS OF USAGE AND ARGUMENTS



def get_args():
    parser = argparse.ArgumentParser("Retinal OCT Classification", add_help=True)
    
    parser.add_argument('--data_dir', type=str, help='Root directory of dataset')
    parser.add_argument('--Name_Exp', type=str, default=None, help='Name of the experiment for logging purposes')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--agg_method', type=str ,default='mean') 
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--k_fold', type=int, default=None, help='Use k-fold cross-validation')
    parser.add_argument('--save_path', type=str, default='./Results')
    
    parser.add_argument('--needed_modalities', nargs='+', default=('sup','deep','cc'), help='List of needed modalities for training (e.g. --needed_modalities sup deep cc perf_map)')
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--run_explainabilty', type= bool, default= False)
    parser.add_argument('--target_class' , type = int , default = None)
    
    return parser.parse_args()

def main(args):
    print("Setting seed...")
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build model
    model = build_model(
        num_classes= args.num_classes,
        input_shape= [args.input_channels, 256, 256],
        pretrained= True,
        adaptive_input= (args.input_channels == 4),
        dropout= args.dropout,
        version= 'b0'
    )
    model.to(device)
    
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    
    # Build data loaders
    if args.k_fold:
        pass
        fold_loaders = get_k_fold_data_loaders(
            data_dir= args.data_dir,
            k=args.k_fold,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            Needed_modalities=args.needed_modalities
        )
        engine = Engine(model, optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
                        loss_fn=torch.nn.CrossEntropyLoss(), data_loader=fold_loaders, K_fold=True, agg_method= args.agg_method, exp_name=args.Name_Exp)
        engine.run_k_fold(epochs=args.epochs, save_path= args.save_path)
    else:
        print("Getting data loaders...")
        train_loader, test_loader = get_dataloader(
            args.data_dir,
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            Needed_modalities=args.needed_modalities
        )
        optimizer=torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        # scheduler=CosineAnnealingWarmRestarts(optimizer, T_0 = 10*len(train_loader))
        engine = Engine(model, optimizer=optimizer,
                        loss_fn=torch.nn.CrossEntropyLoss(reduction="mean"), data_loader=(train_loader, test_loader), agg_method= args.agg_method, exp_name=args.Name_Exp, run_explainability = args.run_explainabilty, XAI_target_class=args.target_class)
        engine.train(epochs=args.epochs,verbose=False, save_path=args.save_path)


if __name__ == "__main__":
    args = get_args()
    main(args)
