from sklearn.model_selection import StratifiedGroupKFold
import torch
import yaml
import os
from torch.utils.data import DataLoader, Subset
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

from src.dataset import ADDataset, create_dataset_csv
from src.model import BERTWithConfidence
from src.training import train_epoch, eval_model, EarlyStopping
from src.utils import set_seed

def main(config):
    set_seed(config['seed'])
    transcription_config = config['transcription']
    model_type = transcription_config['model_type']

    if model_type == 'whisperx':
        model_identifier = transcription_config['whisperx_options']['model_name']
    else: # wav2vec2
        acoustic_id = transcription_config['wav2vec2_options']['acoustic_model_name'].replace('/', '_')
        processor_id = transcription_config['wav2vec2_options']['processor_name'].replace('/', '_')
        model_identifier = f"{acoustic_id}__{processor_id}"

    transcripts_root = os.path.join(transcription_config['output_root'], model_type, model_identifier)
    print(f"Reading transcripts from: {transcripts_root}")

    output_dir = "data/processed"
    test_labels_csv = "data/label_test_task1.csv"
    create_dataset_csv(transcripts_root, output_dir, test_labels_csv, num_hypotheses=config['num_hypotheses'])

    os.makedirs(config['output_dir'], exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading tokenizer: {config['model_name']}")
    tokenizer = BertTokenizer.from_pretrained(config['model_name'])

    print(f"Loading full training dataset...")
    full_train_dataset = ADDataset(config['train_csv'], tokenizer, config['max_len'])
    X = full_train_dataset.df
    y = full_train_dataset.df['label']
    groups = full_train_dataset.df['source_file'] 
    n_splits = config['n_splits']

    if n_splits == 1:
        validation_split_fraction = 0.2
        n_splits_for_single_run = int(1 / validation_split_fraction)
        sgkf = StratifiedGroupKFold(n_splits=n_splits_for_single_run, shuffle=True, random_state=config['seed'])
    else:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config['seed'])

    sgkf_splitter = sgkf.split(X, y, groups)

    for fold in range(n_splits):
        print(f"Fold {fold + 1}/{n_splits}")
        fold_output_dir = os.path.join(config['output_dir'], f"fold_{fold+1}")
        os.makedirs(fold_output_dir, exist_ok=True)

        train_indices, val_indices = next(sgkf_splitter)
        train_subset = Subset(full_train_dataset, train_indices)
        val_subset = Subset(full_train_dataset, val_indices)
        train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
        model = BERTWithConfidence(config['model_name'], num_classes=2).to(device)
    
        # Setup the optimizer and the scheduler
        optimizer = AdamW(model.parameters(), lr=config['learning_rate'], eps=1e-8)
        total_steps = len(train_loader) * config['epochs']
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        loss_fn = torch.nn.CrossEntropyLoss()
        patience = config['patience']
        early_stopping = EarlyStopping(patience=patience, mode='min', min_delta=0.001)
    
        for epoch in range(config['epochs']):
            print(f"\nEpoch {epoch + 1}/{config['epochs']}")
            train_loss = train_epoch(model, train_loader, loss_fn, optimizer, scheduler, device)
            val_loss, val_accuracy = eval_model(model, val_loader, loss_fn, device)
            print(f"Fold {fold+1} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

            if val_loss < early_stopping.best_score:
                model_path = os.path.join(fold_output_dir, "best_model.bin")
                torch.save(model.state_dict(), model_path)
                print(f"Validation loss improved. Model for fold {fold+1} saved.")

            if early_stopping(val_loss):
                print(f"Early stopping activated after {epoch+1} epochs.")
                break   

    print("Cross-validation training finished.")
    print(f"All {n_splits} models have been saved in their respective fold directories.")

if __name__ == "__main__":
    with open("config.yml", 'r') as f:
        config = yaml.safe_load(f)

    transcription_model_id = config['transcription']['model_type']
    classifier_model_name = config['model_name'].replace("/", "_")
    num_hypotheses = config.get('num_hypotheses')
    final_model_folder_name = f"{classifier_model_name}_hypo{num_hypotheses}_from_{transcription_model_id}"
    config['output_dir'] = os.path.join("results/models", final_model_folder_name)
    config['train_csv'] = "data/processed/train.csv"
    
    main(config)