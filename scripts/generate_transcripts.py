import os
import glob
from tqdm import tqdm
import yaml

from src.transcription import transcribe_with_whisperx, transcribe_batch_with_wav2vec2
from src.utils import get_project_paths

def load_whisperx_model(options):
    import whisperx
    """Loads the WhisperX model based on the provided options."""
    print(f"Loading WhisperX model: {options['model_name']} on {options['device']}...")
    model = whisperx.load_model(
        options['model_name'],
        options['device'],
        compute_type=options['compute_type'],
        language="en",
        return_scores=True
    )
    print("WhisperX model loaded.")
    return model

def load_wav2vec2_model(options):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM
    import torch
    """Loads the Wav2Vec2 model and processor based on the provided options."""
    print(f"Loading Wav2Vec2 model from: {options['acoustic_model_name']}")
    device = torch.device(options['device'] if torch.cuda.is_available() else "cpu")
    
    processor = Wav2Vec2ProcessorWithLM.from_pretrained(options['processor_name'])
    model = Wav2Vec2ForCTC.from_pretrained(options['acoustic_model_name']).to(device)
    
    print("Wav2Vec2 model loaded.")
    return model, processor

def main():
    with open("config.yml", 'r') as f:
        config = yaml.safe_load(f)

    paths = get_project_paths(config)
    transcription_config = config['transcription']
    model_type = transcription_config['model_type']

    if model_type == 'whisperx':
        from faster_whisper.transcribe import TranscriptionOptions
        whisper_options = transcription_config['whisperx_options']
        model = load_whisperx_model(whisper_options)
    elif model_type == 'wav2vec2':
        wav2vec_options = transcription_config['wav2vec2_options']
        acoustic_model, processor = load_wav2vec2_model(wav2vec_options)
    else:
        raise ValueError(f"Unsupported model type: {model_type}.")
    
    base_output_dir = paths['transcripts_root']
    print(f"Outputting transcripts to: {base_output_dir}")

    data_sources = {
        "train": config['data']['train_audio_root'],
        "test": config['data']['test_audio_root']
    }
    
    for dataset_type, data_root in data_sources.items():
        output_dir = os.path.join(base_output_dir, dataset_type)
        
        # Find all .wav files
        glob_pattern = os.path.join(data_root, "*/*.wav") if dataset_type == "train" else os.path.join(data_root, "*.wav")
        all_audio_files = glob.glob(glob_pattern)

        # Filter out files that have already been processed to avoid re-transcribing
        audio_files_to_process = []
        for audio_path in all_audio_files:
            relative_path = os.path.relpath(audio_path, data_root)
            output_filename = os.path.splitext(relative_path)[0] + ".json"
            output_path = os.path.join(output_dir, output_filename)
            if not os.path.exists(output_path):
                audio_files_to_process.append((audio_path, output_path))
        
        if not audio_files_to_process:
            print(f"No new files to process for {dataset_type} set ({model_type}).")
            continue

        print(f"Found {len(audio_files_to_process)} new audio files to transcribe for {dataset_type} set...")
        
        if model_type == 'whisperx':
            if dataset_type == "train":
                n_hypo = whisper_options['n_hypotheses']
                model.model.return_nbest = (n_hypo > 1)
                asr_opts = {"best_of": n_hypo, "beam_size": n_hypo}
            else:
                model.model.return_nbest = False
                asr_opts = {"best_of": 5, "beam_size": 5} # Default for whisperX

            # Update the model's transcription options
            current_opts_dict = model.options.__dict__
            current_opts_dict.update(asr_opts)
            model.options = TranscriptionOptions(**current_opts_dict)

            # Process one file at a time for WhisperX
            for audio_path, output_path in tqdm(audio_files_to_process, desc=f"Transcribing (WhisperX) - {dataset_type}"):
                transcribe_with_whisperx(model, audio_path, output_path)
            
        elif model_type == 'wav2vec2':
            batch_size = wav2vec_options['batch_size']
            generate_multiple = True if dataset_type == "train" else False
            # Create batches of files and process them
            for i in tqdm(range(0, len(audio_files_to_process), batch_size), desc=f"Transcribing (Wav2Vec2) - {dataset_type}"):
                batch_tuples = audio_files_to_process[i:i+batch_size]
                
                # Unzip the list of tuples into two separate lists
                audio_batch = [item[0] for item in batch_tuples]
                output_batch = [item[1] for item in batch_tuples]

                transcribe_batch_with_wav2vec2(
                    acoustic_model, 
                    processor, 
                    audio_batch, 
                    output_batch, 
                    wav2vec_options, 
                    generate_multiple_hypotheses=generate_multiple
                )

if __name__ == "__main__":
    main()