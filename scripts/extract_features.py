import torch
import os
import glob
import yaml
import numpy as np
import json
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from collections import defaultdict

from src.utils import load_and_preprocess_audio

def get_pooled_features_from_audio(model, processor, audio_path, device):
    try:
        waveform_tensor = load_and_preprocess_audio(audio_path)

        inputs = processor(waveform_tensor.numpy(), sampling_rate=16000, return_tensors="pt", do_normalize=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        
        hidden_states = outputs.hidden_states
        
        feature_vectors = {}
        for i, layer_hidden_states in enumerate(hidden_states):
            # The shape is (batch, sequence_length, hidden_size). We do a mean-pool over the sequence_length.
            # squeeze(0) removes the batch dimension, resulting in a 1D vector.
            pooled_vector = torch.mean(layer_hidden_states, dim=1).squeeze(0)
            feature_vectors[f'layer_{i}'] = pooled_vector.cpu().numpy()
        
        return feature_vectors

    except Exception as e:
        print(f"Errore durante l'elaborazione di {os.path.basename(audio_path)}: {e}")
        return None

def main():
    with open("config.yml", 'r') as f:
        config = yaml.safe_load(f)

    fe_config = config['feature_extraction']
    model_name = fe_config['model_name']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading feature extraction model: {model_name}")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval()

    model_name_safe = model_name.replace("/", "_")
    output_base_dir = os.path.join(fe_config['output_root'], model_name_safe)
    print(f"Features will be saved to: {output_base_dir}")

    data_sources = {
        "train": config['data']['train_audio_root'],
        "test": config['data']['test_audio_root']
    }
    
    for dataset_type, data_root in data_sources.items():
        print(f"\nProcessing '{dataset_type}' dataset...")
        
        glob_pattern = os.path.join(data_root, "**/*.wav") if dataset_type == "train" else os.path.join(data_root, "*.wav")
        all_audio_files = sorted(glob.glob(glob_pattern, recursive=True)) # Ordina per avere un output consistente
        output_dir = os.path.join(output_base_dir, dataset_type)
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if the job has already been done to allow for resuming
        metadata_path = os.path.join(output_dir, '_metadata.json')
        if os.path.exists(metadata_path):
            print(f"Features for '{dataset_type}' seem to be already extracted. Skipping.")
            continue

        all_features_by_layer = defaultdict(list)
        file_order_metadata = []

        for audio_path in tqdm(all_audio_files, desc=f"Extracting ({dataset_type})"):
            features = get_pooled_features_from_audio(model, processor, audio_path, device)
            if features:
                for layer_name, vector in features.items():
                    all_features_by_layer[layer_name].append(vector)
                file_order_metadata.append(os.path.relpath(audio_path, data_root))
        
        # Save each layer's features as a single .npy matrix
        for layer_name, feature_list in tqdm(all_features_by_layer.items(), desc="Saving layers"):
            feature_matrix = np.vstack(feature_list)
            output_path = os.path.join(output_dir, f"{layer_name}.npy")
            np.save(output_path, feature_matrix)
        
        # Save the metadata (the file order is essential for mapping features back to files/labels)
        with open(metadata_path, 'w') as f:
            json.dump(file_order_metadata, f, indent=2)

    print("\nFeature extraction completed.")

if __name__ == "__main__":
    main()
