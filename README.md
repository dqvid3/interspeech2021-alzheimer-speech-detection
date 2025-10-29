# Alzheimer's Dementia Detection from Spontaneous Speech

This project is a re-implementation and adaptation of methods from the INTERSPEECH 2021 paper: ["Using the Outputs of Different Automatic Speech Recognition Paradigms for Acoustic- and BERT-based Alzheimer's Dementia Detection through Spontaneous Speech"](https://www.isca-archive.org/interspeech_2021/pan21c_interspeech.html) by Pan, Mirheidari, et al.

The goal is to classify speech from the ADReSSo dataset as belonging to individuals with Alzheimer's Disease (AD) or Healthy Controls (HC).

## Implemented Models

This repository faithfully implements all major model architectures described in the paper:

1.  **Acoustic-Only (Model 1):** Extracts features from a pre-trained Wav2Vec2 model and trains traditional classifiers (SVM, RandomForest) to evaluate the predictive power of acoustics alone.
2.  **Linguistic-Only (Model 2):** A standard BERT-based classifier fine-tuned on ASR transcripts.
3.  **Acoustic-Linguistic Fusion (Model 3):** A multi-modal model that fuses linguistic features from BERT and acoustic features from Wav2Vec2 for a combined prediction.
4.  **Linguistic with ASR Confidence (Models 4/5):** An advanced BERT-based model that incorporates ASR confidence scores as an additional input, helping the model to weigh the reliability of the transcribed text.

### Key Differences from the Original Paper

*   **ASR Models**: This project leverages publicly available, state-of-the-art pre-trained models (Wav2Vec2, WhisperX) from Hugging Face, enabling high-quality transcription without the need for custom ASR training.
*   **Flexibility**: The entire pipeline is driven by a central `config.yml` file, making it easy to switch between models, change hyperparameters, and run new experiments.

## Project Structure

-   `/src`: Core Python source code (dataset, models, factories, training logic).
-   `/scripts`: Executable scripts for running each step of the pipeline.
-   `config.yml`: Central configuration file for all hyperparameters and paths.
-   `environment.yml`: Conda environment file for easy setup.

## How to Run

### 1. Setup Environment

This project uses two separate Conda environments to manage dependencies: one for the main pipeline and another for the `whisperx` ASR model, which has specific requirements.

**a) Main Environment:**

First, clone this repository:
```bash
git clone git@github.com:dqvid3/interspeech2021-alzheimer-speech-detection.git
cd interspeech2021-alzheimer-speech-detection
```

Create the main Conda environment using the provided file:
```bash
conda env create -f environment.yml
conda activate ad-detect
```

Finally, install PyTorch with CUDA support matching your GPU. For example, for CUDA 12.1:
```bash
# Check your CUDA version with `nvidia-smi`
pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**b) WhisperX Environment (Optional, if using WhisperX):**

To use the `whisperx` ASR model, you need a separate environment. Please follow the setup instructions on [this forked repository](https://github.com/dqvid3/whisperX), which contains the necessary `environment.yml` and setup guide.

### 2. Download Data

This project uses the **ADReSSo-2021 Challenge dataset**. Access to this dataset is controlled and requires submitting a formal request through the official TalkBank portal.

1.  Visit the [dataset's homepage](https://talkbank.org/dementia/ADReSSo-2021/index.html)
2.  Follow the instructions to request access and download the files.

Once you have obtained the data, create a `data/` directory in the root of this project and place the unzipped files inside, maintaining the following structure:
```
.
├── data/
│   ├── ADReSSo21-diagnosis-train/
│   │   └── ADReSSo21/diagnosis/train/audio/
│   │       ├── ad/
│   │       └── cn/
│   ├── ADReSSo21-diagnosis-test/
│   │   └── ADReSSo21/diagnosis/test-dist/audio/
│   └── label_test_task1.csv
└── src/
└── ... (other project files)
```

### 3. Configuration

Review and edit the **`config.yml`** file to control the entire experiment. You can:
-   Select the `model_type` to train (`text_only`, `confidence`, `fusion`).
-   Choose the ASR model in the `transcription` section (`wav2vec2` or `whisperx`).
-   Adjust hyperparameters, paths, and feature extraction settings.

### 4. Run Pipeline

All scripts must be executed as Python modules from the project's root directory. Ensure the correct Conda environment is activated.

**Step 1: Generate ASR Transcripts**
*(Activate the `whisperx` environment if you've configured it in `config.yml`, otherwise use the main `ad-detect` environment)*
```bash
python -m scripts.generate_transcripts
```

**Step 2 (Optional): Extract and Evaluate Acoustic Features**
*(Activate the `ad-detect` environment)* 
```bash
# Extract features from Wav2Vec2 layers
python -m scripts.extract_features

# Train and evaluate classifiers on these features to find the best layer
python -m scripts.train_classifiers
```
*Note: The best layer found can be set in `config.yml` under `fusion_model.acoustic_feature_layer`.*

**Step 3: Train the Main Classifier**
*(Activate the `ad-detection` environment)*
```bash
# This will train the model specified by `model_type` in config.yml
python -m scripts.train
```

**Step 4: Evaluate on the Test Set**
*(Activate the `ad-detect` environment)*
```bash
# Evaluates the trained model folds and produces an ensemble result
python -m scripts.evaluate
```