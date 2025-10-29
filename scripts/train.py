from sklearn.model_selection import StratifiedGroupKFold
import torch
import yaml
import json
import os
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

from src.training import train_epoch, eval_model, EarlyStopping
from src.utils import set_seed, get_project_paths
from src.factory import build_dataset, build_model, build_tokenizer

def main():
    with open("config.yml", 'r') as f:
        config = yaml.safe_load(f)
    set_seed(config['seed'])

    paths = get_project_paths(config)
    os.makedirs(paths['classifier_output_dir'], exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = build_tokenizer(config)
    full_train_dataset = build_dataset(config, paths, 'train', tokenizer)
    
    X = full_train_dataset.df
    y = full_train_dataset.df['label']
    groups = full_train_dataset.df['source_file'] 
    n_splits = config['n_splits']

    if n_splits == 1:
        n_splits_for_single_run = int(1 / config['validation_split_fraction'])
        sgkf = StratifiedGroupKFold(n_splits=n_splits_for_single_run, shuffle=True, random_state=config['seed'])
    else:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config['seed'])

    sgkf_splitter = sgkf.split(X, y, groups)
    train_config = config['training']

    for fold in range(n_splits):
        print(f"Fold {fold + 1}/{n_splits}")
        fold_output_dir = os.path.join(paths['classifier_output_dir'], f"fold_{fold+1}")
        os.makedirs(fold_output_dir, exist_ok=True)

        train_indices, val_indices = next(sgkf_splitter)
        train_subset = Subset(full_train_dataset, train_indices)
        val_subset = Subset(full_train_dataset, val_indices)
        train_loader = DataLoader(
            train_subset,
            batch_size=train_config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        model = build_model(config, device)

        optimizer = AdamW(model.parameters(), lr=train_config['learning_rate'], eps=1e-8)
        total_steps = len(train_loader) * train_config['epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(total_steps * train_config['warmup_fraction']), 
            num_training_steps=total_steps
        )
        loss_fn = torch.nn.CrossEntropyLoss()
        patience = train_config['patience']
        early_stopping = EarlyStopping(patience=patience, mode='max', min_delta=0.001)
    
        training_history = []
        for epoch in range(train_config['epochs']):
            print(f"\nEpoch {epoch + 1}/{train_config['epochs']}")
            train_loss = train_epoch(model, train_loader, loss_fn, optimizer, scheduler, device, train_config['gradient_clip_norm'])
            val_loss, val_accuracy = eval_model(model, val_loader, device, loss_fn)
            print(f"Fold {fold+1} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

            # Store history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            })

            if val_accuracy > early_stopping.best_score:
                model_path = os.path.join(fold_output_dir, "best_model.bin")
                torch.save(model.state_dict(), model_path)
                print(f"Validation accuracy improved. Model for fold {fold+1} saved.")

            if early_stopping(val_accuracy):
                print(f"Early stopping activated after {epoch+1} epochs.")
                break  

        # Save training history to a JSON file
        history_path = os.path.join(fold_output_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        print(f"Training history for fold {fold+1} saved to {history_path}") 
            
    print("Cross-validation training finished.")

if __name__ == "__main__":
    main()
