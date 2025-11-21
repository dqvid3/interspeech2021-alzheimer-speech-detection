# Alzheimer's Dementia Detection from Spontaneous Speech

This project is a re-implementation and adaptation of methods from the INTERSPEECH 2021 paper: ["Using the Outputs of Different Automatic Speech Recognition Paradigms for Acoustic- and BERT-based Alzheimer's Dementia Detection through Spontaneous Speech"](https://www.isca-archive.org/interspeech_2021/pan21c_interspeech.html) by Pan, Mirheidari, et al.

The goal is to classify speech from the ADReSSo dataset as belonging to individuals with Alzheimer's Disease (AD) or Healthy Controls (HC).

## Implemented Models

This repository faithfully implements all major model architectures described in the paper:

1.  **Acoustic-Only (Model 1):** Extracts features from a pre-trained Wav2Vec2 model and trains traditional classifiers (SVM, RandomForest) to evaluate the predictive power of acoustics alone.
2.  **Linguistic-Only (Model 2):** A standard BERT-based classifier fine-tuned on ASR transcripts.
3.  **Acoustic-Linguistic Fusion (Model 3):** A multi-modal model that fuses linguistic features from BERT and acoustic features from Wav2Vec2 for a combined prediction.
4.  **Linguistic with ASR Confidence (Models 4/5):** An advanced BERT-based model that incorporates ASR confidence scores as an additional input, helping the model to weigh the reliability of the transcribed text.

## Implementation Details & Deviations from Original Paper

While the original paper utilizes Kaldi-based TDNN models and extracts hypothesis paths directly from ASR Lattices, this repository leverages modern Transformer-based ASR architectures (**Wav2Vec2** and **WhisperX**). 

To replicate the **multiple hypotheses generation** required for Models 4 and 5 without access to raw Kaldi lattices, we adopt the following strategies to generate functionally equivalent diverse transcripts:

*   **Wav2Vec2:** We utilize the `pyctcdecode` decoder with a KenLM language model. To generate diversity, we perform a grid sweep over the decoding hyperparameters. The scoring function used to rank hypotheses $y$ is defined as:

    $$ \text{Score}(y) = \log P_{\text{AM}}(y|x) + \alpha \cdot \log P_{\text{LM}}(y) + \beta \cdot |y| $$

    Where:
    *   $\log P_{\text{AM}}$ is the acoustic model probability (Wav2Vec2).
    *   $\log P_{\text{LM}}$ is the language model probability.
    *   $|y|$ is the number of words in the transcript.
    
    We generate 30 distinct hypotheses per audio file by varying:
    *   **$\alpha$ (Language Model Weight):** Controls the contribution of the LM.
    *   **$\beta$ (Word Insertion Penalty):** Penalizes or rewards longer transcriptions.
*   **WhisperX:** We extract the **Top-N** candidates directly from the beam search decoding process.

This approach ensures the BERT classifier receives a rich set of alternative transcriptions and confidence scores, preserving the core methodology of the original study.

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

Finally, install PyTorch and its related packages (torchaudio, torchcodec) with CUDA support matching your GPU. For example, for CUDA 12.1:
```bash
# Check your CUDA version with `nvidia-smi`
pip3 install torch torchaudio torchcodec --index-url https://download.pytorch.org/whl/cu121
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
*(Activate the `ad-detect` environment)*
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

## Citation

If you use this code or methodology, please cite the original paper:

```bibtex
@inproceedings{pan21c_interspeech,
  title     = {Using the Outputs of Different Automatic Speech Recognition Paradigms for Acoustic- and BERT-Based Alzheimer’s Dementia Detection Through Spontaneous Speech},
  author    = {Yilin Pan and Bahman Mirheidari and Jennifer M. Harris and Jennifer C. Thompson and Matthew Jones and Julie S. Snowden and Daniel Blackburn and Heidi Christensen},
  year      = {2021},
  booktitle = {Interspeech 2021},
  pages     = {3810--3814},
  doi       = {10.21437/Interspeech.2021-1519},
  issn      = {2958-1796},
}