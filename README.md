# Alzheimer's Dementia Detection from Spontaneous Speech

This project is a re-implementation and adaptation of methods from the INTERSPEECH 2021 paper: ["Using the Outputs of Different Automatic Speech Recognition Paradigms for Acoustic- and BERT-based Alzheimer's Dementia Detection through Spontaneous Speech"](https://www.isca-archive.org/interspeech_2021/pan21c_interspeech.html) by Pan, Mirheidari, et al.

The goal is to classify speech from the ADReSSo dataset as belonging to individuals with Alzheimer's Disease (AD) or Healthy Controls (HC).

## Current Implementation

This repository currently implements a linguistic-based classification pipeline inspired by **Models 4/5** from the paper. The key components are:

1.  **Automatic Speech Recognition (ASR)**: Scripts to transcribe audio files using modern, pre-trained ASR models (e.g., Wav2Vec2, Whisper). The system can generate multiple hypotheses per audio file to be used as a form of data augmentation.
2.  **BERT-based Classification**: A `BERT-large` model that takes the transcribed text and ASR confidence scores as input to perform the final AD vs. HC classification.
3.  **Cross-Validation**: A robust training and evaluation setup using Stratified Group K-Fold cross-validation to ensure that samples from the same speaker do not appear in both the training and validation sets of a given fold.

### Key Differences from the Original Paper

*   **ASR Models**: This project leverages publicly available pre-trained models from Hugging Face and OpenAI, instead of the custom-trained models used in the original research.
*   **Focus**: The current implementation focuses exclusively on the linguistic classification model. Acoustic-only and fusion models are potential future extensions.

## Project Structure

-   `/src`: Core Python source code for the dataset, model, and training logic.
-   `/scripts`: Executable scripts for running the pipeline.
-   `config.yml`: Central configuration file for hyperparameters and paths.

## How to Run

### 1. Setup

First, clone this repository to your local machine:
```bash
git clone https://github.com/dqvid3/interspeech2021-alzheimer-speech-detection.git
cd interspeech2021-alzheimer-speech-detection
```

It is highly recommended to create a dedicated Python environment for this project using Conda or venv.

```bash
# Example using Conda
conda create --name ad-detection python=3.10
conda activate ad-detection
```

Next, install the required dependencies. An `environment.yml` file will be provided.

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

Before running the pipeline, review and edit the **`config.yml`** file. Here you can set hyperparameters, choose the ASR model, and define other settings for your experiments.

### 4. Run Pipeline

All scripts must be executed as Python modules from the project's root directory to ensure correct imports.

```bash
# 1. Generate ASR transcripts for the train and test sets
# This will read audio files from `data/` and save transcripts to `results/`
python -m scripts.generate_transcripts

# 2. Train the classification model using cross-validation
# This reads the generated transcripts and trains a model for each fold
python -m scripts.train

# 3. Evaluate the trained models on the test set
# This performs inference and calculates final performance metrics
python -m scripts.evaluate
```