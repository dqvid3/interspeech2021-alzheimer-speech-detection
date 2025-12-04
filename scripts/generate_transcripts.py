import os
import glob
from tqdm import tqdm

from src.transcription import transcribe_with_whisperx, transcribe_batch_with_wav2vec2
from src.utils import setup_experiment

def load_whisperx_model(options, device):
    """Loads the WhisperX model based on the provided options."""
    import whisperx
    from huggingface_hub import snapshot_download

    print(f"Loading WhisperX model: {options['model_name']} on {device}...")
    if options['model_name'] != 'large-v3':
        model_path = snapshot_download(repo_id=options['model_name'])
    else:
        model_path = options['model_name']
    
    model = whisperx.load_model(
        model_path,
        device=device.type,
        compute_type=options['compute_type'],
        language=options['language'],
        return_scores=True
    )
    print("WhisperX model loaded.")
    return model

def load_wav2vec2_model(options, device):
    """
    Loads the Wav2Vec2 model and dynamically builds the processor with a custom KenLM model.
    """
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM
    from pyctcdecode import build_ctcdecoder
    from huggingface_hub import hf_hub_download

    print(f"Loading Wav2Vec2 model from: {options['model_name']}")
    acoustic_model = Wav2Vec2ForCTC.from_pretrained(options['model_name']).to(device)

    print(f"Loading base processor from: {options['model_name']}")
    base_processor = Wav2Vec2Processor.from_pretrained(options['model_name'])
    
    lm_repo_id = options['lm_repo_id']
    lm_filename = options['lm_filename']
    print(f"Downloading Language Model '{lm_filename}' from '{lm_repo_id}'...")
    lm_path = hf_hub_download(repo_id=lm_repo_id, filename=lm_filename)
    
    print("Building CTC decoder with the new Language Model...")
    vocab_dict = base_processor.tokenizer.get_vocab()
    sorted_vocab_list = sorted(vocab_dict.items(), key=lambda item: item[1])
    vocab = [x[0] for x in sorted_vocab_list]

    unigram_list = None
    if options.get('lm_unigrams_filename'):
        unigrams_filename = options['lm_unigrams_filename']
        print(f"Downloading unigrams '{unigrams_filename}' from '{lm_repo_id}'...")
        unigrams_path = hf_hub_download(
            repo_id=lm_repo_id, 
            filename=unigrams_filename
        )
        with open(unigrams_path, 'r', encoding='utf-8') as f:
            unigram_list = [line.strip() for line in f]

    decoder = build_ctcdecoder(
        labels=vocab,
        kenlm_model_path=lm_path,
        unigrams=unigram_list
    )
    
    processor = Wav2Vec2ProcessorWithLM(
        feature_extractor=base_processor.feature_extractor,
        tokenizer=base_processor.tokenizer,
        decoder=decoder
    )
    
    print("Wav2Vec2 model and custom LM processor loaded successfully.")
    return acoustic_model, processor

def main():
    config, paths, device = setup_experiment()
    transcription_config = config['transcription']
    model_type = transcription_config['model_type']

    if model_type == 'whisperx':
        from faster_whisper.transcribe import TranscriptionOptions
        whisper_options = transcription_config['whisperx_options']
        model = load_whisperx_model(whisper_options, device)
    elif model_type == 'wav2vec2':
        wav2vec_options = transcription_config['wav2vec2_options']
        acoustic_model, processor = load_wav2vec2_model(wav2vec_options, device)
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
                transcribe_with_whisperx(model, audio_path, output_path, batch_size=whisper_options['batch_size'])
            
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
