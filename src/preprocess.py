import pandas as pd
import numpy as np
import time
import argparse
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
import onnxruntime as ort

def load_onnx_model(max_length=256):
    print("[*] Downloading INT8 Quantized ONNX weights natively from HuggingFace (Zero PyTorch)...")
    # Pulling from Xenova's pre-exported repository which provides strict `.onnx` binaries directly
    model_path = hf_hub_download(repo_id="Xenova/all-MiniLM-L6-v2", filename="onnx/model_quantized.onnx")
    tokenizer_path = hf_hub_download(repo_id="Xenova/all-MiniLM-L6-v2", filename="tokenizer.json")
    
    # Initialize extremely fast Rust tokenizer engine
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.enable_padding(length=max_length)
    
    # Initialize ONNX Execution Engine strictly mapping to PC CPU cores
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Keep threading conservative to reduce lockups on some Windows setups.
    sess_options.intra_op_num_threads = max(1, min(8, os.cpu_count() or 4))
    sess_options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    return tokenizer, session

def mean_pooling(token_embeddings, attention_mask):
    # Maps complex multi-token structures down into a single flattened 384-dimensional dense vector representing the abstract
    mask_expanded = np.expand_dims(attention_mask, -1)
    sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
    sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask

def run_preprocessing(batch_size=128, max_length=192, limit=None):
    root_dir = Path(__file__).parent.parent
    processed_dir = root_dir / 'data' / 'processed'
    
    print("\n[*] Phase 2: Native ONNX CPU Embedding Evaluation")
    
    # Step 1: Load and Validate Data
    df = pd.read_csv(processed_dir / 'arxiv_cleaned.csv')
    
    if df['cleaned_abstract'].isnull().any():
        df.dropna(subset=['cleaned_abstract'], inplace=True)
        
    texts = df['cleaned_abstract'].astype(str).tolist()
    
    # Encode categorical textual labels natively into mathematical integer matrices
    labels = df['category'].astype("category").cat.codes.values
    classes = df['category'].astype("category").cat.categories.values

    if limit is not None:
        limit = max(1, int(limit))
        texts = texts[:limit]
        labels = labels[:limit]
        print(f"    -> Running in limited mode on first {limit} rows")
    
    print(f"    -> Validated {len(texts)} unique sequences mapping to {len(classes)} logical classes")
    
    # Step 2: Optimal Embedding Model
    tokenizer, session = load_onnx_model(max_length=max_length)
    
    # Step 3: Fast CPU Mathematical Batch Iteration
    embeddings = []
    
    print(f"[=] Firing batch generation arrays via pure CPU execution. Batch Size: {batch_size}")
    start_t = time.time()
    
    total = len(texts)
    total_batches = (total + batch_size - 1) // batch_size

    for batch_idx, i in enumerate(range(0, total, batch_size), start=1):
        batch_texts = texts[i:i+batch_size]
        
        encodings = tokenizer.encode_batch(batch_texts)
        
        # Map Rust representations into explicit NumPy C array contiguous structures
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)
        
        # Execute absolute ONNX forwards pass physically bypassing Python memory loops
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }
        
        ort_outs = session.run(None, ort_inputs)
        token_embeddings = ort_outs[0]
        
        # Mathematically pool and export natively
        sentence_embeddings = mean_pooling(token_embeddings, attention_mask)
        embeddings.append(sentence_embeddings)
        
        if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == total_batches:
            elapsed = time.time() - start_t
            done = min(i + len(batch_texts), total)
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (total - done) / rate if rate > 0 else 0.0
            print(
                f"    -> Batch {batch_idx}/{total_batches} | "
                f"rows {done}/{total} | "
                f"{rate:.1f} rows/sec | ETA {eta/60:.1f} min"
            )
            
    X = np.vstack(embeddings)
    print(f"[+] Materialized {len(X)} vector profiles perfectly in {time.time()-start_t:.2f} seconds.")
    
    # Step 4: System Serialization Checkpoints
    np.save(processed_dir / 'X.npy', X)
    np.save(processed_dir / 'y.npy', labels)
    np.save(processed_dir / 'classes.npy', classes)
    
    print("[DONE] Artifacts firmly saved successfully! Ready for Phase 3 MLFlow Evaluation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ONNX embeddings and label arrays.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for ONNX inference.")
    parser.add_argument("--max-length", type=int, default=192, help="Tokenizer max length.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick test runs.")
    args = parser.parse_args()
    run_preprocessing(batch_size=args.batch_size, max_length=args.max_length, limit=args.limit)
