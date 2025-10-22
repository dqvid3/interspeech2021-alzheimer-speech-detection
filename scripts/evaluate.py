import torch
import os
import numpy as np
import yaml
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import mode
from tqdm import tqdm

from src.dataset import ADDataset, create_dataset_csv
from src.model import BERTWithConfidence
from src.utils import set_seed, get_project_paths

def get_predictions_and_probs(model, data_loader, device):
    model.eval()

    all_preds, all_probs = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            confidence_score = batch['confidence_score'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                confidence_score=confidence_score
            )
            
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return all_preds, all_probs

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open("config.yml", 'r') as f:
        config = yaml.safe_load(f)
    set_seed(config['seed'])

    paths = get_project_paths(config)

    print(f"Reading test transcripts from: {paths['transcripts_root']}")
    
    create_dataset_csv(
        transcripts_root=paths['transcripts_root'],
        output_dir=paths['processed_data_dir'],
        test_labels_path=config['data']['test_labels_csv'],
        num_hypotheses=None
    )
    
    model_dir = paths['classifier_output_dir']
    print(f"Loading models from: {model_dir}")

    fold_dirs = sorted([d for d in os.listdir(model_dir) if d.startswith("fold_")])

    print(f"Loading tokenizer: {config['model_name']}")
    tokenizer = BertTokenizer.from_pretrained(config['model_name'])

    print(f"Loading test data from: {paths['test_csv']}")
    test_dataset = ADDataset(paths['test_csv'], tokenizer, config['max_len'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    true_labels = test_dataset.df['label'].values
    all_fold_predictions = []
    all_fold_probabilities = []

    for fold_dir in tqdm(fold_dirs, desc="Evaluating folds"):
        model_path = os.path.join(model_dir, fold_dir, "best_model.bin")
        
        model = BERTWithConfidence(config['model_name'], num_classes=2).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        predictions, probabilities = get_predictions_and_probs(model, test_loader, device)
        all_fold_predictions.append(predictions)
        all_fold_probabilities.append(probabilities)

    fold_accuracies = [accuracy_score(true_labels, preds) for preds in all_fold_predictions]
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    print(f"Number of folds evaluated: {len(fold_accuracies)}")
    for i, acc in enumerate(fold_accuracies):
        print(f"  - Fold {i+1} Accuracy: {acc:.4f}")
    
    print(f"\nMean Accuracy: {mean_acc:.4f}")
    print(f"Standard Deviation of Accuracy: {std_acc:.4f}")

    predictions_array = np.array(all_fold_predictions).T # (n_folds, n_samples) -> (n_samples, n_folds)
    ensemble_preds_vote, _ = mode(predictions_array, axis=1, keepdims=False)
    ensemble_accuracy_vote = accuracy_score(true_labels, ensemble_preds_vote)
    print(f"Ensemble Accuracy (Majority Vote): {ensemble_accuracy_vote:.4f}")

    mean_probs = np.mean(all_fold_probabilities, axis=0)
    ensemble_preds_avg = np.argmax(mean_probs, axis=1)
    ensemble_accuracy_avg = accuracy_score(true_labels, ensemble_preds_avg)
    print(f"Ensemble Accuracy (Probability Averaging): {ensemble_accuracy_avg:.4f}\n")
    
    print("Classification Report for Ensemble (Majority Vote)")
    print(classification_report(true_labels, ensemble_preds_vote, target_names=['Control', 'ProbableAD']))
    
    print("Classification Report for Ensemble (Probability Averaging)")
    print(classification_report(true_labels, ensemble_preds_avg, target_names=['Control', 'ProbableAD']))