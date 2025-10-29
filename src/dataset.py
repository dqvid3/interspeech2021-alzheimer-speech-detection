import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import re
import os
import glob
import json
import math
import numpy as np

class ADDataset(Dataset):
    """
    Custom PyTorch Dataset for the Alzheimer's Dementia detection task.
    Reads a CSV file containing text, confidence scores, and labels.
    Can also load pre-extracted acoustic features for fusion models.
    """
    def __init__(self, csv_path, tokenizer, max_len, acoustic_features_path=None, acoustic_metadata_path=None):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.acoustic_features = None
        self.file_to_feature_idx = None

        if acoustic_features_path and acoustic_metadata_path:
            print(f"Loading acoustic features from: {acoustic_features_path}")
            self.acoustic_features = np.load(acoustic_features_path)
            
            with open(acoustic_metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.file_to_feature_idx = {os.path.basename(path): i for i, path in enumerate(metadata)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        text = str(row.text)
        score = row.confidence_score
        label = row.label
        source_file = row.source_file

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'confidence_score': torch.tensor(score, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long)
        }
        
        if self.acoustic_features is not None:
            feature_idx = self.file_to_feature_idx.get(source_file)
            acoustic_feature = self.acoustic_features[feature_idx]
            item['acoustic_feature'] = torch.tensor(acoustic_feature, dtype=torch.float)

        return item
    
def create_dataset_csv(transcripts_root, output_dir, test_labels_path=None, num_hypotheses=None):
    """
    Reads JSON files with transcription hypotheses and creates separate CSV files 
    for training and/or testing.

    For the 'train' set, labels are inferred from the directory structure (ad/cn).
    For the 'test' set, labels are loaded from a provided CSV file.

    Args:
        transcripts_root (str): Path to the main directory of transcripts (e.g., 'results/transcripts').
        output_dir (str): Directory where the CSV files will be saved.
        test_labels_path (str): Path to the CSV file containing labels for the test set.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if test_labels_path is not None:
        dataset_type = "test"
    else:
        dataset_type = "train"

    print(f"Processing {dataset_type} set...")
    
    data_path = os.path.join(transcripts_root, dataset_type)
    all_examples = []

    if dataset_type == "test":
        labels_df = pd.read_csv(test_labels_path)
        # Create a dictionary for fast lookup: {'adrsdt15': 0, 'adrsdt46': 1}
        label_map = {
            row['ID']: 1 if row['Dx'] == 'ProbableAD' else 0 
            for _, row in labels_df.iterrows()
        }

    # Search for all JSON files in the subdirectories (ad, cn)
    glob_pattern = os.path.join(data_path, "*/*.json") if dataset_type == "train" else os.path.join(data_path, "*.json")
    json_files = glob.glob(glob_pattern)

    for json_file in tqdm(json_files, desc=f"Reading {dataset_type} JSONs"):
        source_filename = os.path.basename(json_file).replace('.json', '.wav')
        
        if dataset_type == "train":
            # For train set, infer from folder structure
            label = 1 if os.path.basename(os.path.dirname(json_file)) == 'ad' else 0
        else:
            # Look up the label in the map for test set
            file_id = os.path.splitext(os.path.basename(json_file))[0]
            label = label_map[file_id]

        with open(json_file, 'r', encoding='utf-8') as f:
            hypotheses = json.load(f)
        
        if num_hypotheses is not None:
            hypotheses_to_process = hypotheses[:num_hypotheses]
        else:
            hypotheses_to_process = hypotheses

        for hypo in hypotheses_to_process:
            text = hypo.get("text", "")

            text_orig = text
            
            # Normalize disfluency annotations
            text = re.sub(r'\[\s*uh\s*\]', 'uh', text)
            text = re.sub(r'\[\s*um\s*\]', 'um', text)

            avg_log_score = hypo.get("avg_log_score", -100.0)             
            # Convert log-score to a confidence score in the [0, 1] range
            confidence_score = math.exp(avg_log_score)

            all_examples.append({
                "text": text,
                "confidence_score": confidence_score,
                "label": label,
                "source_file": source_filename
            })

    # Create a DataFrame and save it as a CSV file
    df = pd.DataFrame(all_examples)
    output_path = os.path.join(output_dir, f"{dataset_type}.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved {dataset_type} dataset to {output_path}. Total examples: {len(df)}")
