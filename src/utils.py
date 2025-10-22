import random
import numpy as np
import torch
import os

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # for multi-GPU
        # These settings may slow down training, but are
        # necessary for full reproducibility on cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set an environment variable for reproducibility of certain operations
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    print(f"Seed set to: {seed_value}")

def get_project_paths(config):
    """
    Generates all necessary project paths from the configuration dictionary.
    This centralizes path management and ensures consistency across scripts.
    """
    transcription_config = config['transcription']
    asr_model_type = transcription_config['model_type']
    
    if asr_model_type == 'whisperx':
        asr_model_id = transcription_config['whisperx_options']['model_name']
    else: # wav2vec2
        acoustic_id = transcription_config['wav2vec2_options']['acoustic_model_name'].replace('/', '_')
        processor_id = transcription_config['wav2vec2_options']['processor_name'].replace('/', '_')
        asr_model_id = f"{acoustic_id}__{processor_id}"

    classifier_model_name = config['model_name'].replace("/", "_")
    num_hypotheses = config.get('num_hypotheses')
    classifier_folder_name = f"{classifier_model_name}_hypo{num_hypotheses}_from_{asr_model_type}"

    results_root = config['results_root']
    processed_data_dir = config['data']['processed_data_dir']

    paths = {
        'transcripts_root': os.path.join(results_root, "transcripts", asr_model_type, asr_model_id),
        'classifier_output_dir': os.path.join(results_root, "models", classifier_folder_name),
        'train_csv': os.path.join(processed_data_dir, "train.csv"),
        'test_csv': os.path.join(processed_data_dir, "test.csv"),
        'processed_data_dir': processed_data_dir
    }
    return paths