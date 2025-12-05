from transformers import AutoTokenizer

from src.dataset import ADDataset, create_dataset_csv
from src.model import ADBERTClassifier
from src.utils import get_acoustic_feature_paths

def build_tokenizer(config):
    """Factory function to build and return a tokenizer."""
    print(f"Loading tokenizer: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    return tokenizer

def build_dataset(config, paths, dataset_type, tokenizer):
    """Factory function to build and return a dataset (train or test)."""
    model_type = config['model_type']
    print(f"Reading test transcripts from: {paths['transcripts_root']}")
    if dataset_type == 'train' and model_type == 'confidence':
        num_hypotheses = config['num_hypotheses']
    else:
        num_hypotheses = 1

    create_dataset_csv(
        transcripts_root=paths['transcripts_root'],
        output_dir=paths['processed_data_dir'],
        test_labels_path=config['data']['test_labels_csv'] if dataset_type == 'test' else None,
        num_hypotheses=num_hypotheses,
        num_classes=config['num_classes']
    )

    csv_path = paths['train_csv'] if dataset_type == 'train' else paths['test_csv']
    
    dataset_args = {
        'csv_path': csv_path,
        'tokenizer': tokenizer,
        'max_len': config['max_len']
    }

    if model_type == 'fusion':
        features_path, metadata_path = get_acoustic_feature_paths(config, dataset_type)
        dataset_args['acoustic_features_path'] = features_path
        dataset_args['acoustic_metadata_path'] = metadata_path

    print(f"Loading {dataset_type} dataset from: {csv_path}")
    return ADDataset(**dataset_args)

def build_model(config, device):
    """Factory function to build and return the correct model based on config."""
    model = ADBERTClassifier(config)
    return model.to(device)
