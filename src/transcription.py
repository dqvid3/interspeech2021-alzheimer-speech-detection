import json
import os
import numpy as np
import torch

from src.utils import load_and_preprocess_audio

def save_hypotheses_to_json(hypotheses_list, output_path):
    """
    Helper function to sort hypotheses and save them to a JSON file.
    """
    # Sort hypotheses by avg_log_score in descending order (best first)
    hypotheses_list.sort(key=lambda x: x['avg_log_score'], reverse=True)
    
    # Create the final list in the required format, adding the index
    final_output = []
    for index, hypo in enumerate(hypotheses_list):
        final_output.append({
            "hypothesis_index": index,
            "text": hypo["text"],
            "avg_log_score": hypo["avg_log_score"]
        })
        
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

def transcribe_with_whisperx(model, audio_file, output_path, batch_size):
    """
    Transcribes a single audio file using WhisperX and saves the hypotheses.
    """
    try:
        result = model.transcribe(audio_file, batch_size=batch_size)
    except Exception as e:
        print(f"Error during WhisperX transcription of {audio_file}: {e}")
        return

    full_hypotheses = []
    
    # Check if n-best hypotheses were generated
    has_nbest = model.model.return_nbest and len(result["segments"]) > 0 and "hypotheses" in result["segments"][0]
    num_hypotheses = len(result["segments"][0]["hypotheses"]) if has_nbest else 1

    for h in range(num_hypotheses):
        full_text = ""
        total_score = 0.0
        segment_count = len(result["segments"])
        
        for seg in result["segments"]:
            if h == 0:
                # The first hypothesis is always the main "text" field
                full_text += " " + seg["text"]
                total_score += seg.get("score", 0.0)
            elif has_nbest:
                hypo = seg["hypotheses"][h]
                full_text += " " + hypo["text"]
                total_score += hypo.get("score", 0.0)
        
        # The score is a log-probability; we average it over the segments
        avg_score = total_score / segment_count if segment_count > 0 else 0.0
        
        full_hypotheses.append({
            "text": full_text.strip(),
            "avg_log_score": avg_score
        })
    
    save_hypotheses_to_json(full_hypotheses, output_path)

def transcribe_batch_with_wav2vec2(acoustic_model, processor, audio_files_batch, output_paths_batch, options, generate_multiple_hypotheses=True):
    """
    Transcribes a batch of audio files using Wav2Vec2 with an LM and saves the hypotheses.
    """
    try:
        raw_audio_list = [load_and_preprocess_audio(f).numpy() for f in audio_files_batch]
        device = acoustic_model.device

        inputs = processor(raw_audio_list, return_tensors="pt", padding=True, sampling_rate=16000).to(device)
        with torch.no_grad():
            logits = acoustic_model(**inputs).logits

        # Dictionary to accumulate hypotheses for each file in the batch
        results_by_file = {path: [] for path in output_paths_batch}

        def process_decoded_transcriptions(transcriptions):
            for i, output_path in enumerate(output_paths_batch):
                transcription_text = transcriptions.text[i].lower()
                logit_score = transcriptions.logit_score[i]
                
                num_characters = len(transcription_text)
                avg_log_score = logit_score / num_characters if num_characters > 0 else -100.0
                
                hypothesis = {
                    "text": transcription_text,
                    "avg_log_score": avg_log_score,
                }
                results_by_file[output_path].append(hypothesis)
        
        if generate_multiple_hypotheses:
            alphas = np.linspace(options['min_alpha'], options['max_alpha'], options['num_alphas'])
            betas = np.linspace(options['min_beta'], options['max_beta'], options['num_betas'])
            # Generate hypotheses for the entire batch at once
            for alpha in alphas:
                for beta in betas:
                    decoded = processor.batch_decode(logits.cpu().numpy(), alpha=alpha, beta=beta)
                    process_decoded_transcriptions(decoded)
        else:
            # For the test set, generate only the single best hypothesis
            decoded = processor.batch_decode(logits.cpu().numpy())
            process_decoded_transcriptions(decoded)

        # Save the results for each file
        for output_path, hypotheses_list in results_by_file.items():
            save_hypotheses_to_json(hypotheses_list, output_path)

    except Exception as e:
        print(f"Error during Wav2Vec2 batch transcription for {len(audio_files_batch)} files: {e}")
        return
