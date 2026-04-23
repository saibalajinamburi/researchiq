import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import re

STOP_WORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", 
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", 
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", 
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", 
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", 
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", 
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", 
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
])

def strip_latex(text: str) -> str:
    """Scientific papers heavily abuse LaTeX formatting in abstracts which destroys embeddings."""
    text = re.sub(r'\$.*?\$', ' ', text)
    text = re.sub(r'\\math[a-z]+', ' ', text)
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)
    return text

def clean_text(text: str) -> str:
    """Converts text to strictly lowercase ascii alphanumerics free of LaTeX pollution and stop words."""
    text = text.lower()
    text = strip_latex(text)
    text = re.sub(r'[^a-z0-9#\+]', ' ', text)
    words = text.split()
    cleaned_words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    return " ".join(cleaned_words)

def run_eda_pipeline():
    root_dir = Path(__file__).parent.parent
    raw_csv = root_dir / 'data' / 'raw' / 'arxiv_50k.csv'
    processed_dir = root_dir / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    figures_dir = root_dir / 'reports' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Loading Original ArXiv Dataset...")
    df = pd.read_csv(raw_csv)
    original_len = len(df)
    
    # Clean Duplicates
    df.drop_duplicates(subset=['title'], inplace=True)
    print(f"[-] Dropped {original_len - len(df)} duplicate papers. Total unique: {len(df)}")
    
    # Text Preprocessing 
    print("[*] Applying NLP Stop-Word & Regex cleaning to abstract texts...")
    start_t = time.time()
    df['cleaned_abstract'] = df['abstract'].apply(clean_text)
    print(f"[+] Text cleaned in {time.time() - start_t:.2f} seconds.")
    
    # Generating EDA Feature: Word Counts
    df['word_count'] = df['cleaned_abstract'].apply(lambda x: len(x.split()))
    
    # Save Processed Checkpoint
    df.to_csv(processed_dir / 'arxiv_cleaned.csv', index=False)
    print(f"[+] Saved Cleaned Dataset to data/processed/arxiv_cleaned.csv")
    
    # GRAPH GENERATION
    print("\n[*] Generating Explanatory Data Graphs...")
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(10, 6))
    class_counts = df['category'].value_counts()
    sns.barplot(x=class_counts.values, y=class_counts.index, hue=class_counts.index, palette="viridis", legend=False)
    plt.title("ArXiv Extracted Document Class Distribution", fontsize=14, weight='bold')
    plt.xlabel("Number of Unique Papers")
    plt.ylabel("Academic Category")
    plt.tight_layout()
    plt.savefig(figures_dir / 'class_distribution.png', dpi=150)
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='category', y='word_count', hue='category', palette="magma", legend=False)
    plt.title("Word Count Variance Across Academic Domains (Cleaned)", fontsize=14, weight='bold')
    plt.xlabel("Academic Category")
    plt.ylabel("Semantic Words per Abstract")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(figures_dir / 'abstract_length_dist.png', dpi=150)
    
    print("\n[DONE] EDA & Cleaning Pipeline Complete!")

if __name__ == "__main__":
    run_eda_pipeline()
