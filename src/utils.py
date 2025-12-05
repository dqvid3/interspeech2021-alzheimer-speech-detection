import random
import numpy as np
import torch
import yaml
import torchaudio
import os

def _set_seed(seed_value=42):
    """Sets the seed for generating random numbers to ensure reproducibility."""
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
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
        torch.use_deterministic_algorithms(True)

    # Set an environment variable for reproducibility of certain operations
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    print(f"Seed set to: {seed_value}")

def load_and_preprocess_audio(audio_path):
    """
    Loads an audio file, resamples it to 16kHz, and converts it to mono.

    Args:
        audio_path (str): The path to the audio file.

    Returns:
        torch.Tensor: The preprocessed waveform as a 1D tensor.
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample to 16kHz if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    # Convert to mono by averaging channels if the audio is stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Squeeze to remove the channel dimension and return a 1D tensor
    return waveform.squeeze()

def get_acoustic_feature_paths(config, dataset_type):
    """Generates paths for acoustic features and metadata based on the configuration."""
    fe_config = config['feature_extraction']
    model_name_safe = fe_config['model_name'].replace("/", "_")
    features_base_dir = os.path.join(fe_config['output_root'], model_name_safe)
    layer_file = config['fusion_model']['acoustic_feature_layer'] + ".npy"
    
    features_path = os.path.join(features_base_dir, dataset_type, layer_file)
    metadata_path = os.path.join(features_base_dir, dataset_type, '_metadata.json')
    return features_path, metadata_path

def _get_project_paths(config):
    """
    Generates all necessary project paths from the configuration dictionary.
    This centralizes path management and ensures consistency across scripts.
    """
    transcription_config = config['transcription']
    asr_model_type = transcription_config['model_type']
    
    if asr_model_type == 'whisperx':
        asr_model_id = transcription_config['whisperx_options']['model_name'].replace('/', '_')
    else: # wav2vec2
        wav2vec_opts = transcription_config['wav2vec2_options']
        acoustic_id = wav2vec_opts['model_name'].replace('/', '_')
        if 'lm_repo_id' in wav2vec_opts and wav2vec_opts['lm_repo_id']:
            lm_id = wav2vec_opts['lm_repo_id'].replace('/', '_')
            asr_model_id = f"{acoustic_id}_LM_{lm_id}"
        else:
            asr_model_id = acoustic_id

    classifier_model_name = config['model_name']
    num_hypotheses = config['num_hypotheses']
    model_type_classifier = config['model_type']
    hypo_str = f"_hypo{num_hypotheses}" if model_type_classifier == 'confidence' else ""
    # Include the specific ASR model ID in the classifier folder name for clarity
    classifier_folder_name = f"{classifier_model_name}_{model_type_classifier}{hypo_str}_from_{asr_model_type}_{asr_model_id}"

    results_root = config['results_root']
    processed_data_dir = config['data']['processed_data_dir']

    paths = {
        'transcripts_root': os.path.join(results_root, "transcripts", asr_model_type, asr_model_id),
        'classifier_output_dir': os.path.join(results_root, "models", classifier_folder_name),
        'train_csv': os.path.join(processed_data_dir, "train.csv"),
        'test_csv': os.path.join(processed_data_dir, "test.csv"),
        'processed_data_dir': processed_data_dir,
        'asr_model_type': asr_model_type,
        'asr_model_id': asr_model_id
    }
    return paths

def setup_experiment():
    with open("config.yml", 'r') as f:
        config = yaml.safe_load(f)

    _set_seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if config['model_type'] == 'confidence':
        config['model_name'] = config['model_name_large']
    else:
        config['model_name'] = config['model_name_base']
    
    paths = _get_project_paths(config)
    
    return config, paths, device

def get_label(label):
    s = str(label).lower()

    if 'mci' in s:
        return 2
    elif 'ad' in s:
        return 1
    
    return 0 # Control
