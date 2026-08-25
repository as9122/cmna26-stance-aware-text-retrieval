import os
import torch
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from sentence_transformers import util
from rank_bm25 import BM25Okapi

# Define available datasets
DATASET_KEYS = [
    "averitec", 
    "tfu_evaluation", 
    "tfu_training_heldout", 
    "argtumour", 
    "tfu_validation_gpt5", 
    "tfu_validation_qwen"
]

# Argument parsing
parser = argparse.ArgumentParser(description="Evaluate Precomputed Embeddings")
parser.add_argument("--input_file", type=str, default=None, 
                    help="Path to a specific .pt precomputed vector file to evaluate.")
parser.add_argument("--input_dir", type=str, default=None, 
                    help="Path to a directory containing .pt files. If provided, evaluates all files in the directory.")
parser.add_argument("--k", type=int, default=10, 
                    help="The cutoff threshold (K) for ranking metrics. Default is 10.")
parser.add_argument("--lambdas", nargs="+", type=float, default=[0.1, 0.15, 0.2, 0.3, 0.4, 0.5], 
                    help="List of lambda weights to test for Reciprocal Score Fusion (RSF).")
args = parser.parse_args()

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
PRECOMPUTED_DIR = os.path.join(RESULTS_DIR, "precomputed_vectors")

# Helper Functions
def calculate_metrics(retrieved_indices, corpus_meta, target_topic, target_intent, cutoff_k):
    R = sum(1 for doc in corpus_meta if any(r['topic'] == target_topic and r['label'] == target_intent for r in doc['relations']))
    if R == 0: return None, 0.0, 0.0, 0.0, 0.0, 0, {}
    
    retrieved_relevance = []
    rank_labels_out = {} 
    
    for i, idx in enumerate(retrieved_indices):
        relations = [(r['topic'], r['label']) for r in corpus_meta[idx]['relations']]
        is_relevant = (target_topic, target_intent) in relations
        retrieved_relevance.append(1 if is_relevant else 0)
        
        topics_in_doc = [r[0] for r in relations]
        if target_topic in topics_in_doc:
            doc_stance = "support" if (target_topic, "support") in relations else "attack"
            rank_labels_out[f"Rank_{i+1}"] = doc_stance
        else:
            intents_in_doc = [r[1] for r in relations]
            rank_labels_out[f"Rank_{i+1}"] = "Irrelevant / Correct Stance" if target_intent in intents_in_doc else "Irrelevant / Incorrect Stance"

    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(retrieved_relevance))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(retrieved_relevance), R)))
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    cutoff = min(R, len(retrieved_indices))
    correct_top_R = stance_err_R = irr_correct_R = irr_incorrect_R = 0

    for i in range(cutoff):
        relations = [(r['topic'], r['label']) for r in corpus_meta[retrieved_indices[i]]['relations']]
        topics_in_doc = [r[0] for r in relations]
        intents_in_doc = [r[1] for r in relations]
        
        if target_topic in topics_in_doc:
            if (target_topic, target_intent) in relations: correct_top_R += 1
            else: stance_err_R += 1
        else:
            if target_intent in intents_in_doc: irr_correct_R += 1
            else: irr_incorrect_R += 1

    precision_at_R = correct_top_R / cutoff if cutoff > 0 else 0.0
    stance_error_at_R = stance_err_R / cutoff if cutoff > 0 else 0.0
    irr_correct_at_R = irr_correct_R / cutoff if cutoff > 0 else 0.0
    irr_incorrect_at_R = irr_incorrect_R / cutoff if cutoff > 0 else 0.0
    
    return ndcg, precision_at_R, stance_error_at_R, irr_correct_at_R, irr_incorrect_at_R, R, rank_labels_out

def get_evaluated_models(csv_path):
    if not os.path.exists(csv_path): return set()
    try:
        df = pd.read_csv(csv_path, usecols=["Model"])
        return set(df["Model"].unique())
    except: return set()

def run_evaluation():
    if not args.input_file and not args.input_dir:
        print(f"No explicit input provided. Scanning default directory: {PRECOMPUTED_DIR}")
        args.input_dir = PRECOMPUTED_DIR

    pt_files = []
    if args.input_file:
        if os.path.exists(args.input_file):
            pt_files.append(args.input_file)
        else:
            print(f"File not found: {args.input_file}")
            return
    
    if args.input_dir:
        if os.path.exists(args.input_dir):
            pt_files.extend([os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.pt')])
        else:
            print(f"Directory not found: {args.input_dir}")
            return

    if not pt_files:
        print("No .pt files found to process.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for file_path in pt_files:
        print(f"\nLoading package: {file_path}")
        package = torch.load(file_path, map_location="cpu", weights_only=False)
        dataset_id = package.get("dataset")
        model_name = package.get("model_name", "Unknown_Model")
        
        if dataset_id not in DATASET_KEYS:
            print(f"Warning: Dataset ID '{dataset_id}' not recognized in DATASET_KEYS. Skipping.")
            continue

        out_dir = os.path.join(RESULTS_DIR, f"eval_{dataset_id}")
        os.makedirs(out_dir, exist_ok=True)
        
        res_csv_path = os.path.join(out_dir, "experiment_results.csv")
        rank_csv_path = os.path.join(out_dir, "ranking_patterns.csv")
        evaluated_models = get_evaluated_models(res_csv_path)

        # Build list of all outputs we expect to generate for this model
        expected_outputs = [model_name]
        for lam in args.lambdas:
            expected_outputs.append(f"{model_name}_Hybrid_RSF_{lam}")

        # Find exactly which runs are missing from the CSV
        missing_runs = [out for out in expected_outputs if out not in evaluated_models]

        # If everything is already in the CSV, skip processing this .pt file
        if not missing_runs:
            print(f"--- Skipping: {model_name} already fully evaluated on {dataset_id.upper()} ---")
            continue

        need_dense = model_name in missing_runs
        need_hybrid = any("Hybrid" in out for out in missing_runs)

        print(f"\n==========================================")
        print(f"Evaluating {model_name} on {dataset_id.upper()}")
        print(f"Pending Runs to Execute:")
        if need_dense:
            print(f"  -> Dense Search ({model_name})")
        for lam in args.lambdas:
            if f"{model_name}_Hybrid_RSF_{lam}" in missing_runs:
                print(f"  -> Hybrid Search (RSF, λ = {lam})")
        print(f"==========================================")

        corpus_meta = package["corpus_metadata"]
        query_meta = package["query_metadata"]
        corpus_embs = package["corpus_embs"].to(device).float()
        query_embs = package["query_embs"].to(device).float()

        # Initialise BM25 only if we need hybrid runs
        bm25 = None
        if need_hybrid:
            tokenized_corpus = [doc["lexical_tokens"] for doc in corpus_meta]
            bm25 = BM25Okapi(tokenized_corpus)

        current_res = []
        current_rank = []
        bm25_score_cache = {}

        for i, meta in enumerate(tqdm(query_meta, desc="Evaluating Queries")):
            q_emb = query_embs[i].unsqueeze(0)
            
            dense_scores = util.cos_sim(q_emb, corpus_embs)[0].cpu().numpy()
            
            # Dense Evaluation
            if need_dense:
                dense_top_indices = np.argsort(dense_scores, kind='stable')[::-1][:args.k]
                ndcg, prec_R, err_R, irr_corr_R, irr_incorr_R, r_count, rank_labels = calculate_metrics(
                    dense_top_indices, corpus_meta, meta["topic"], meta["intent"], args.k
                )
                
                if ndcg is not None:
                    current_res.append({
                        "Model": model_name, "Topic": meta["topic"], "Intent": meta["intent"], 
                        "Instruction": meta["instruction"], "Instruction_Type": meta["instruction_type"],
                        f"NDCG@{args.k}": ndcg, "Precision@R": prec_R, "Stance_Error_R": err_R,
                        "Irrelevant_Correct_Stance_R": irr_corr_R, "Irrelevant_Incorrect_Stance_R": irr_incorr_R
                    })
                    rank_row = {"Model": model_name, "Topic": meta["topic"], "Intent": meta["intent"], "Instruction": meta["instruction"], "Instruction_Type": meta["instruction_type"], "Total_Relevant": r_count}
                    rank_row.update(rank_labels)
                    rank_row.update({f"Idx_{j+1}": idx for j, idx in enumerate(dense_top_indices)})
                    current_rank.append(rank_row)

            # Hybrid Evaluation
            if need_hybrid:
                topic_key = meta["topic"]
                
                if topic_key not in bm25_score_cache:
                    bm25_score_cache[topic_key] = np.array(bm25.get_scores(meta["lexical_tokens"]))
                
                bm25_scores = bm25_score_cache[topic_key]
                
                # RSF Strategy (Min-Max Normalization)
                norm_dense = (dense_scores - np.min(dense_scores)) / (np.ptp(dense_scores) + 1e-8)
                norm_bm25 = (bm25_scores - np.min(bm25_scores)) / (np.ptp(bm25_scores) + 1e-8)

                for lam in args.lambdas:
                    name_rsf = f"{model_name}_Hybrid_RSF_{lam}"
                    if name_rsf in missing_runs:
                        fused_scores = norm_dense + (float(lam) * norm_bm25)
                        
                        rsf_sorted_indices = np.argsort(fused_scores, kind='stable')[::-1]
                        rsf_top = rsf_sorted_indices[:args.k]
                        
                        ndcg, prec_R, err_R, irr_corr_R, irr_incorr_R, r_count, rank_labels = calculate_metrics(
                            rsf_top, corpus_meta, meta["topic"], meta["intent"], args.k
                        )
                        if ndcg is not None:
                            current_res.append({
                                "Model": name_rsf, "Topic": meta["topic"], "Intent": meta["intent"], "Instruction": meta["instruction"], "Instruction_Type": meta["instruction_type"],
                                f"NDCG@{args.k}": ndcg, "Precision@R": prec_R, "Stance_Error_R": err_R, "Irrelevant_Correct_Stance_R": irr_corr_R, "Irrelevant_Incorrect_Stance_R": irr_incorr_R
                            })
                            rank_row = {"Model": name_rsf, "Topic": meta["topic"], "Intent": meta["intent"], "Instruction": meta["instruction"], "Instruction_Type": meta["instruction_type"], "Total_Relevant": r_count}
                            rank_row.update(rank_labels)
                            rank_row.update({f"Idx_{j+1}": idx for j, idx in enumerate(rsf_top)})
                            current_rank.append(rank_row)

        # Save Checkpoint
        if current_res:
            pd.DataFrame(current_res).to_csv(res_csv_path, mode='a', header=not os.path.exists(res_csv_path), index=False)
            pd.DataFrame(current_rank).to_csv(rank_csv_path, mode='a', header=not os.path.exists(rank_csv_path), index=False)
            print(f"Results successfully saved to {dataset_id} CSVs.")

    print("\nAll local evaluations complete!")

if __name__ == "__main__":
    run_evaluation()