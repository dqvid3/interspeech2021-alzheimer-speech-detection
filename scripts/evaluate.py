import json
import torch
import os
import numpy as np
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import mode
from tqdm import tqdm

from src.training import eval_model
from src.utils import set_seed, get_project_paths
from src.factory import build_dataset, build_model, build_tokenizer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open("config.yml", 'r') as f:
        config = yaml.safe_load(f)
    set_seed(config['seed'])

    paths = get_project_paths(config)
    train_config = config['training']
    
    model_dir = paths['classifier_output_dir']
    print(f"Loading models from: {model_dir}")

    fold_dirs = sorted([d for d in os.listdir(model_dir) if d.startswith("fold_")])

    tokenizer = build_tokenizer(config)
    test_dataset = build_dataset(config, paths, 'test', tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=train_config['batch_size'], shuffle=False, num_workers=8, pin_memory=True)

    true_labels = test_dataset.df['label'].values
    all_fold_predictions = []
    all_fold_probabilities = []

    for fold_dir in tqdm(fold_dirs, desc="Evaluating folds"):
        model_path = os.path.join(model_dir, fold_dir, "best_model.bin")

        model = build_model(config, device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        predictions, probabilities = eval_model(model, test_loader, device)
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

    # Save final evaluation report
    final_report = {
        "mean_accuracy": mean_acc,
        "std_dev_accuracy": std_acc,
        "fold_accuracies": {f"fold_{i+1}": acc for i, acc in enumerate(fold_accuracies)},
        "ensemble_accuracy_majority_vote": ensemble_accuracy_vote,
        "ensemble_accuracy_probability_averaging": ensemble_accuracy_avg,
        "classification_report_majority_vote": classification_report(
            true_labels, ensemble_preds_vote, target_names=['Control', 'ProbableAD'], output_dict=True
        ),
        "classification_report_probability_averaging": classification_report(
            true_labels, ensemble_preds_avg, target_names=['Control', 'ProbableAD'], output_dict=True
        )
    }

    report_path = os.path.join(model_dir, "evaluation_report.json")
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print(f"\nFinal evaluation report saved to: {report_path}")
