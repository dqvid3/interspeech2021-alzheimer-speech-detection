import numpy as np
import pandas as pd
import json
import os
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline   
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.stats import mode
from tqdm import tqdm

def main():
    with open("config.yml", 'r') as f:
        config = yaml.safe_load(f)

    fe_config = config['feature_extraction']
    model_name = fe_config['model_name']
    model_name_safe = model_name.replace("/", "_")
    
    # Path to the directory where features are stored
    features_base_dir = os.path.join(fe_config['output_root'], model_name_safe)

    classifier_configs = {
        'RandomForest': {
            'pipeline': Pipeline([
                ('classifier', RandomForestClassifier(n_estimators=10, n_jobs=-1))
            ])
        },
        'SVM': {
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(probability=True))
            ])
        }
    }

    # Load training data and labels
    train_features_dir = os.path.join(features_base_dir, 'train')
    train_metadata_path = os.path.join(train_features_dir, '_metadata.json')
    if not os.path.exists(train_metadata_path):
        print(f"Error: Training metadata not found at {train_metadata_path}. Run feature extraction first.")
        return
    with open(train_metadata_path, 'r') as f:
        train_metadata = json.load(f)
    y_train_full = np.array([1 if 'ad/' in path or 'ad\\' in path else 0 for path in train_metadata])

    # Load test data and labels
    test_features_dir = os.path.join(features_base_dir, 'test')
    test_metadata_path = os.path.join(test_features_dir, '_metadata.json')
    if not os.path.exists(test_metadata_path):
        print(f"Error: Test metadata not found at {test_metadata_path}. Run feature extraction first.")
        return
    with open(test_metadata_path, 'r') as f:
        test_metadata = json.load(f)

    # Load true labels for the test set from the provided CSV
    test_labels_df = pd.read_csv(config['data']['test_labels_csv'])
    # Create a mapping from ID (e.g., 'adrsdt001') to Label (0 or 1)
    label_map = {row['ID']: 1 if row['Dx'] == 'ProbableAD' else 0 for _, row in test_labels_df.iterrows()}
    # Get the ID from the metadata path and look up the label
    test_file_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_metadata]
    y_test_full = np.array([label_map[id] for id in test_file_ids])

    # Find all layer feature files
    layer_files = sorted([f for f in os.listdir(train_features_dir) if f.endswith('.npy')])
    if not layer_files:
        print(f"Error: No .npy feature files found in {train_features_dir}")
        return
    print(f"Found {len(layer_files)} feature layers to evaluate.")
    
    all_results = []

    for layer_file in tqdm(layer_files, desc="Evaluating Layers"):
        layer_name = os.path.splitext(layer_file)[0]
        
        # Load training and test features for the current layer
        X_train_full = np.load(os.path.join(train_features_dir, layer_file))
        X_test_full = np.load(os.path.join(test_features_dir, layer_file))

        for clf_name, clf_config in classifier_configs.items():
            
            n_splits = 10
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config['seed'])
            
            cv_fold_accuracies = []
            trained_pipelines = []

            for fold_idx, (train_index, val_index) in enumerate(skf.split(X_train_full, y_train_full)):
                X_train, X_val = X_train_full[train_index], X_train_full[val_index]
                y_train, y_val = y_train_full[train_index], y_train_full[val_index]

                # Train the model for this fold
                pipeline = clone(clf_config['pipeline'])
                if 'classifier__random_state' in pipeline.get_params().keys():
                    pipeline.set_params(classifier__random_state=config['seed'] + fold_idx)

                pipeline.fit(X_train, y_train)

                # Evaluate on the validation set for this fold
                val_preds = pipeline.predict(X_val)
                acc = accuracy_score(y_val, val_preds)
                cv_fold_accuracies.append(acc)
                
                # Save the trained model
                trained_pipelines.append(pipeline)
        
            mean_cv_accuracy = np.mean(cv_fold_accuracies)
            
            all_test_predictions = []
            all_test_probabilities = []

            for pipeline in trained_pipelines:
                # Each of the 10 models predicts on the entire test set
                test_preds = pipeline.predict(X_test_full)
                test_probs = pipeline.predict_proba(X_test_full)
                all_test_predictions.append(test_preds)
                all_test_probabilities.append(test_probs)

            # Calculate accuracy for each of the 10 models on the test set
            test_fold_accuracies = [accuracy_score(y_test_full, preds) for preds in all_test_predictions]
            mean_test_accuracy = np.mean(test_fold_accuracies)

            # Ensemble: Majority Vote
            # Stack predictions: (10 models, num_test_samples)
            predictions_array = np.array(all_test_predictions)
            ensemble_preds_vote, _ = mode(predictions_array, axis=0, keepdims=False)
            ensemble_accuracy_vote = accuracy_score(y_test_full, ensemble_preds_vote)
            prec_vote, rec_vote, f1_vote, _ = precision_recall_fscore_support(y_test_full, ensemble_preds_vote, average=None)

            # Ensemble: Probability Averaging
            # Stack probabilities: (10 models, num_test_samples, 2 classes)
            probabilities_array = np.array(all_test_probabilities)
            mean_probs = np.mean(probabilities_array, axis=0)
            ensemble_preds_avg = np.argmax(mean_probs, axis=1)
            ensemble_accuracy_avg = accuracy_score(y_test_full, ensemble_preds_avg)
            prec_avg, rec_avg, f1_avg, _ = precision_recall_fscore_support(y_test_full, ensemble_preds_avg, average=None)
            
            layer_results = {
                'classifier': clf_name,
                'layer': layer_name,
                'mean_cv_accuracy': mean_cv_accuracy,
                'mean_test_accuracy': mean_test_accuracy,

                'test_ensemble_vote_acc': ensemble_accuracy_vote,
                'test_ensemble_vote_precision_control': prec_vote[0],
                'test_ensemble_vote_recall_control': rec_vote[0],
                'test_ensemble_vote_f1_control': f1_vote[0],
                'test_ensemble_vote_precision_ad': prec_vote[1],
                'test_ensemble_vote_recall_ad': rec_vote[1],
                'test_ensemble_vote_f1_ad': f1_vote[1],

                'test_ensemble_prob_avg_acc': ensemble_accuracy_avg,
                'test_ensemble_prob_avg_precision_control': prec_avg[0],
                'test_ensemble_prob_avg_recall_control': rec_avg[0],
                'test_ensemble_prob_avg_f1_control': f1_avg[0],
                'test_ensemble_prob_avg_precision_ad': prec_avg[1],
                'test_ensemble_prob_avg_recall_ad': rec_avg[1],
                'test_ensemble_prob_avg_f1_ad': f1_avg[1],

                'cv_fold_accuracies': cv_fold_accuracies,       # Accuracy of every fold on val set
                'test_fold_accuracies': test_fold_accuracies   # Accuracy of every fold on test set
            }
            all_results.append(layer_results)

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by='test_ensemble_vote_acc', ascending=False)
    
    print("\nFinal Test Set Evaluation Summary (Sorted by Ensemble Accuracy)")
    print(results_df[[
        'layer',
        'mean_cv_accuracy',         
        'mean_test_accuracy',     
        'test_ensemble_vote_acc',
        'test_ensemble_prob_avg_acc',
    ]].to_string(index=False))
    
    output_dir = "results/classifier_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    results_filename = os.path.join(output_dir, f"classifier_final_evaluation_{model_name_safe}.csv")
    results_df.to_csv(results_filename, index=False)
    print(f"\nDetailed results saved to: {results_filename}")

    results_df['layer_num'] = results_df['layer'].str.split('_').str[1].astype(int)

    for clf_name in results_df['classifier'].unique():
        print(f"\nGenerating plot for: {clf_name}...")
        
        # Filter the DataFrame to get results only for the current classifier
        clf_df = results_df[results_df['classifier'] == clf_name].sort_values('layer_num').reset_index(drop=True)
        
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'sans-serif'
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot each accuracy metric for the current classifier
        ax.plot(
            clf_df['layer_num'], 
            clf_df['mean_cv_accuracy'], 
            label='Mean CV Accuracy (on Validation Folds)', 
            marker='o', 
            linestyle='--',
            color='green'
        )
        ax.plot(
            clf_df['layer_num'], 
            clf_df['test_ensemble_vote_acc'], 
            label='Ensemble Accuracy on Test Set (Majority Vote)', 
            marker='s',
            linestyle='-',
            color='blue'
        )
        ax.plot(
            clf_df['layer_num'], 
            clf_df['test_ensemble_prob_avg_acc'], 
            label='Ensemble Accuracy on Test Set (Probability Avg)', 
            marker='^',
            linestyle='-',
            color='red'
        )
        
        ax.set_title(f'{clf_name} Performance by Wav2Vec2 Layer\n(Acoustic Model: {model_name})', fontsize=16, fontweight='bold')
        ax.set_xlabel("Wav2Vec2 Layer Number", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_xticks(clf_df['layer_num'])
        
        # Adjust y-axis limits dynamically for the current classifier's data
        min_acc = clf_df[['mean_cv_accuracy', 'test_ensemble_vote_acc', 'test_ensemble_prob_avg_acc']].min().min()
        max_acc = clf_df[['mean_cv_accuracy', 'test_ensemble_vote_acc', 'test_ensemble_prob_avg_acc']].max().max()
        ax.set_ylim(min_acc - 0.05, max_acc + 0.05)

        ax.legend(fontsize=11)
        fig.tight_layout()

        plot_dir = os.path.join(output_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        plot_filename = os.path.join(plot_dir, f"{clf_name}_layer_performance_{model_name_safe}.png")
        plt.savefig(plot_filename, dpi=300)
        
        print(f"Plot saved successfully to: '{plot_filename}'")

if __name__ == "__main__":
    main()
