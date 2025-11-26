import pandas as pd
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

from src.utils import setup_experiment

def analyze_token_lengths(config, paths):
    """
    Analyzes and visualizes the token length distributions for train and test sets separately.
    The recommendation for max_len is based strictly on the training set distribution.
    
    Args:
        model_name (str): The name of the tokenizer model to use.
        train_csv_path (str): Path to the training data CSV.
        test_csv_path (str): Path to the test data CSV.
    """
    print("Loading datasets...")
    try:
        model_name = config['model_name']
        train_csv_path = paths['train_csv']
        test_csv_path = paths['test_csv']
        df_train = pd.read_csv(train_csv_path)
        df_test = pd.read_csv(test_csv_path)
        train_texts = df_train['text'].dropna().astype(str)
        test_texts = df_test['text'].dropna().astype(str)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure CSV files are in the 'data/processed/' directory.")
        return

    print(f"Loading tokenizer: {model_name}...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    print("Calculating token lengths for the training set...")
    train_token_lengths = [len(tokenizer.encode(text)) for text in tqdm(train_texts)]
    
    print("Calculating token lengths for the test set...")
    test_token_lengths = [len(tokenizer.encode(text)) for text in tqdm(test_texts)]
    
    # Analyze Distributions and Recommend max_len based on Training data
    train_series = pd.Series(train_token_lengths)
    test_series = pd.Series(test_token_lengths)

    print("\nTraining Set Statistics")
    train_stats = train_series.describe(percentiles=[0.90, 0.95, 0.96, 0.97, 0.99])
    print(train_stats)

    print("\nTest Set Statistics")
    print(test_series.describe())

    model_hard_limit = tokenizer.model_max_length
    recommended_len = int(train_stats['95%'])
    final_max_len = min(recommended_len, model_hard_limit)
    
    print(f"Model's absolute max length: {model_hard_limit}")
    print(f"Based on the training set, 95% of transcripts have a length of {recommended_len} or less.")

    print("\nGenerating comparative plot...")
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))
    
    # Plot both histograms with transparency to see overlaps
    sns.histplot(train_token_lengths, color='blue', label='Train Set', kde=True, alpha=0.6)
    sns.histplot(test_token_lengths, color='green', label='Test Set', kde=True, alpha=0.6)
    
    plt.axvline(final_max_len, color='r', linestyle='dashed', linewidth=2, 
                label=f"Recommended max_len = {final_max_len} (from Train 95th percentile)")
                
    plt.title("Comparative Distribution of Token Lengths")
    plt.xlabel("Token Length")
    plt.ylabel("Frequency")
    plt.legend()
    
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/token_length_distribution_comparative.svg")
    print("Plot saved to 'results/figures/token_length_distribution_comparative.svg'")
    
    plt.show()

if __name__ == "__main__":
    config, paths, _ = setup_experiment()
    
    analyze_token_lengths(config, paths)
