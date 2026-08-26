import os
import gc
import json
import torch
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from peft import PeftModel
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from huggingface_hub import hf_hub_download

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
parser = argparse.ArgumentParser(description="Evaluate Embeddings with Reranker")
parser.add_argument("--base_model", type=str, required=True, choices=["BGE-Large", "Instructor-XL", "Qwen3-Embedding-8B"], 
                    help="The base foundation model architecture to use.")
parser.add_argument("--adapter", type=str, default=None, 
                    help="Optional: Hugging Face Hub ID or local directory path to a trained LoRA adapter.")
parser.add_argument("--reranker", type=str, default="Qwen/Qwen3-Reranker-8B", 
                    help="The CrossEncoder reranker model to use (default: Qwen/Qwen3-Reranker-8B).")
parser.add_argument("--datasets", nargs="+", choices=DATASET_KEYS, default=DATASET_KEYS, 
                    help="Which datasets to evaluate. If omitted, runs all available tracks.")
parser.add_argument("--local_data_dir", type=str, default=None, 
                    help="Optional: Local directory path containing the evaluation JSON corpora. If omitted, downloads from HF.")
parser.add_argument("--run_name", type=str, default=None, 
                    help="Optional: Identifier for this run. If omitted, an automatic name is generated.")
parser.add_argument("--top_n", type=int, default=100, 
                    help="Number of documents to retrieve via dense search before reranking (default: 100).")
parser.add_argument("--k", type=int, default=10, 
                    help="Final cutoff for metrics calculation (default: 10).")
parser.add_argument("--batch_size", type=int, default=64, 
                    help="Batch size for dense encoding (default: 64).")
args = parser.parse_args()

# Determine the model display name
if args.run_name:
    MODEL_RUN_NAME = args.run_name
else:
    MODEL_RUN_NAME = args.base_model if not args.adapter else f"{args.base_model}_{os.path.basename(args.adapter.strip('/'))}"

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")

# Prompt formatters & Base Models
def format_instruct(instruction_text, topic_text):
    return f"Instruct: {instruction_text}\nQuery: {topic_text}"

def format_instructor(instruction_text, topic_text):
    return f"{instruction_text} {topic_text}"

BASE_MODELS = {
    "BGE-Large": {"path": "BAAI/bge-large-en-v1.5", "formatter": format_instruct},
    "Instructor-XL": {"path": "hkunlp/instructor-xl", "formatter": format_instructor},
    "Qwen3-Embedding-8B": {"path": "Qwen/Qwen3-Embedding-8B", "formatter": format_instruct}
}

INSTRUCTIONS = {
    "support": [
        "Discover sources promoting", "Retrieve evidence confirming",
        "Locate specific arguments justifying", "Find statements endorsing",
        "Show points backing up"
    ],
    "attack": [
        "Discover sources dismissing", "Retrieve evidence disproving",
        "Locate specific arguments attacking", "Find statements rejecting",
        "Show points doubting"
    ]
}

INSTRUCTION_POOL = []
for intent in ["support", "attack"]:
    for instr in INSTRUCTIONS[intent]:
        INSTRUCTION_POOL.append({"intent": intent, "text": instr, "type": "Unseen"})

def load_model_helper(base_arch, adapter_path):
    config = BASE_MODELS[base_arch]
    base_path = config["path"]
    
    print(f"Loading Base Model: {base_path}...")
    model = SentenceTransformer(base_path, trust_remote_code=True, model_kwargs={"torch_dtype": torch.bfloat16})
    
    if adapter_path:
        print(f"Injecting PEFT adapter from: {adapter_path}...")
        peft_model = PeftModel.from_pretrained(model[0].auto_model, adapter_path)
        model[0].auto_model = peft_model.merge_and_unload()
        
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, config["formatter"]

def calculate_metrics(retrieved_indices, corpus, target_topic, target_intent):
    R = sum(1 for doc in corpus if any(r['topic'] == target_topic and r['label'] == target_intent for r in doc['relations']))
    if R == 0: return None, 0.0, 0.0, 0.0, 0.0, 0
    
    retrieved_relevance = []
    
    for i, idx in enumerate(retrieved_indices):
        relations = [(r['topic'], r['label']) for r in corpus[idx]['relations']]
        is_relevant = (target_topic, target_intent) in relations
        retrieved_relevance.append(1 if is_relevant else 0)

    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(retrieved_relevance))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(retrieved_relevance), R)))
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    cutoff = min(R, len(retrieved_indices))
    correct_top_R = stance_err_R = irr_correct_R = irr_incorrect_R = 0

    for i in range(cutoff):
        relations = [(r['topic'], r['label']) for r in corpus[retrieved_indices[i]]['relations']]
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
    
    return ndcg, precision_at_R, stance_error_at_R, irr_correct_at_R, irr_incorrect_at_R, R

def run_evaluation():
    print(f"Loading Global Reranker: {args.reranker}...")
    reranker = CrossEncoder(
        args.reranker, 
        trust_remote_code=True, 
        model_kwargs={"torch_dtype": torch.bfloat16}
    )
    
    print(f"\nInitializing execution for: {MODEL_RUN_NAME}")
    model, formatter = load_model_helper(args.base_model, args.adapter)
    
    for track_id in args.datasets:
        filename = f"{track_id}.json"
        print(f"\n{'*'*50}\nSTARTING TRACK: {track_id.upper()}\n{'*'*50}")

        out_dir = os.path.join(RESULTS_DIR, f"eval_{track_id}")
        os.makedirs(out_dir, exist_ok=True)
        res_csv_path = os.path.join(out_dir, "experiment_results_reranked.csv")
        
        if args.local_data_dir and os.path.exists(args.local_data_dir):
            corpus_path = os.path.join(args.local_data_dir, filename)
            print(f"Loading {filename} from local directory: {args.local_data_dir}...")
        else:
            repo_name = "as9122/stance-evaluation-corpora"
            print(f"Downloading {filename} from Hugging Face: {repo_name}...")
            corpus_path = hf_hub_download(repo_id=repo_name, filename=filename, repo_type="dataset")
            
        with open(corpus_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        topic_lookup = {t["topic_id"]: t["topic"] for t in data["topics"]}
        corpus = []
        for doc in data["documents"]:
            hydrated_relations = [{"topic": topic_lookup.get(r["topic_id"], ""), "label": r["label"]} for r in doc["relations"]]
            corpus.append({"text": doc["text"], "relations": hydrated_relations})
            
        corpus_texts = [doc['text'] for doc in corpus]
        unique_topics = list(set([rel['topic'] for doc in corpus for rel in doc['relations']]))

        print("Encoding corpus (batched)...")
        corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=True, batch_size=args.batch_size)
        
        print("Preparing queries...")
        all_queries = []
        query_metadata = []
        for topic in unique_topics:
            for item in INSTRUCTION_POOL:
                full_query_text = formatter(item["text"], topic)
                all_queries.append(full_query_text)
                query_metadata.append({
                    "topic": topic, "intent": item["intent"], 
                    "instruction": item["text"], "type": item["type"]
                })

        print("Encoding queries (batched)...")
        query_embeddings = model.encode(all_queries, convert_to_tensor=True, show_progress_bar=True, batch_size=args.batch_size)
        
        current_res = []
        completed_queries = set()
        
        # Read existing progress to allow resuming
        if os.path.exists(res_csv_path):
            try:
                df_existing = pd.read_csv(res_csv_path)
                target_model = MODEL_RUN_NAME + "_Reranked"
                df_model = df_existing[df_existing["Model"] == target_model]
                for _, row in df_model.iterrows():
                    completed_queries.add((row["Topic"], row["Intent"], row["Instruction"]))
            except Exception as e:
                print(f"Could not read existing CSV for resuming: {e}")

        SAVE_EVERY = 50

        for i, meta in enumerate(tqdm(query_metadata, desc="Retrieving & Reranking Metrics")):
            query_signature = (meta["topic"], meta["intent"], meta["instruction"])
            if query_signature in completed_queries:
                continue 

            q_emb = query_embeddings[i].unsqueeze(0)
            
            # STAGE 1: Initial Retrieval (Top N)
            cos_scores = util.cos_sim(q_emb, corpus_embeddings)[0]
            retrieve_n = min(args.top_n, len(corpus_texts))
            
            top_n_results = torch.topk(cos_scores, k=retrieve_n)
            top_n_indices = top_n_results.indices.cpu().numpy()
            
            # STAGE 2: Cross-Encoder Reranking (Top K)
            query_text = all_queries[i]
            pairs = [[query_text, corpus_texts[idx]] for idx in top_n_indices]
            
            rerank_scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
            
            reranked_relative_indices = np.argsort(rerank_scores, kind='stable')[::-1]
            
            final_k = min(args.k, len(reranked_relative_indices))
            top_k_relative_indices = reranked_relative_indices[:final_k]
            
            top_indices = [top_n_indices[idx] for idx in top_k_relative_indices]
            
            # STAGE 3: Metrics
            metrics = calculate_metrics(
                top_indices, corpus, target_topic=meta["topic"], target_intent=meta["intent"]
            )
            ndcg, prec_R, err_R, irr_corr_R, irr_incorr_R = metrics

            if ndcg is not None:
                current_res.append({
                    "Model": MODEL_RUN_NAME + "_Reranked", "Topic": meta["topic"], "Intent": meta["intent"], 
                    "Instruction": meta["instruction"], "Instruction_Type": meta["type"],
                    f"NDCG@{args.k}": ndcg, "Precision@R": prec_R, "Stance_Error_R": err_R,
                    "Irrelevant_Correct_Stance_R": irr_corr_R, "Irrelevant_Incorrect_Stance_R": irr_incorr_R
                })

            # Periodic Disk Flushing
            if (i + 1) % SAVE_EVERY == 0 and current_res:
                pd.DataFrame(current_res).to_csv(res_csv_path, mode='a', header=not os.path.exists(res_csv_path), index=False)
                current_res.clear()

        # Final flush for any remaining queries
        if current_res:
            pd.DataFrame(current_res).to_csv(res_csv_path, mode='a', header=not os.path.exists(res_csv_path), index=False)
            print(f"Completed {MODEL_RUN_NAME} (Reranked) results to {track_id} CSVs.")

        del corpus_embeddings
        del query_embeddings
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    print("\nAll evaluations complete!")

if __name__ == "__main__":
    run_evaluation()