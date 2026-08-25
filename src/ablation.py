import os
import json
import torch
import pandas as pd
import gc
import argparse
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from peft import PeftModel
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
parser = argparse.ArgumentParser(description="Run word ablation analysis")
parser.add_argument("--base_model", type=str, required=True, choices=["BGE-Large", "Instructor-XL", "Qwen3-Embedding-8B"], 
                    help="The base foundation model architecture to use.")
parser.add_argument("--adapter", type=str, default=None, 
                    help="Optional: Hugging Face Hub ID (e.g., 'as9122/qwen3-8b-embedding-stance-mixed') or a local directory path (e.g., './results/finetuned_models/Qwen3-8B_Mixed_final') to a trained LoRA adapter.")
parser.add_argument("--dataset", type=str, required=True, choices=DATASET_KEYS, 
                    help="Which dataset to evaluate.")
parser.add_argument("--local_data_dir", type=str, default=None, 
                    help="Optional: Local directory path containing the evaluation JSON corpora. If omitted, downloads from Hugging Face.")
parser.add_argument("--run_name", type=str, default=None, 
                    help="Optional: Identifier for this run in the CSV (e.g., 'BGE-Large_Mixed'). If omitted, an automatic name is generated.")
args = parser.parse_args()

# Determine the model display name for the CSV
if args.run_name:
    MODEL_RUN_NAME = args.run_name
else:
    MODEL_RUN_NAME = args.base_model if not args.adapter else f"{args.base_model}_{os.path.basename(args.adapter.strip('/'))}"

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")

DATASET_NAMES = {
    "averitec": "AVeriTeC Arguments",
    "tfu_evaluation": "TFU Evaluation Arguments",
    "tfu_training_heldout": "TFU Training Arguments (20% held-out)",
    "argtumour": "ArgTumour Arguments",
    "tfu_validation_gpt5": "TFU Validation Arguments (GPT-5)",
    "tfu_validation_qwen": "TFU Validation Arguments (Qwen3-8B)"
}

# Prompt formatters and base models
def format_instruct(instruction_text, topic_text):
    return f"Instruct: {instruction_text}\nQuery: {topic_text}"

def format_instructor(instruction_text, topic_text):
    return f"{instruction_text} {topic_text}"

BASE_MODELS = {
    "BGE-Large": {"path": "BAAI/bge-large-en-v1.5", "formatter": format_instruct},
    "Instructor-XL": {"path": "hkunlp/instructor-xl", "formatter": format_instructor},
    "Qwen3-Embedding-8B": {"path": "Qwen/Qwen3-Embedding-8B", "formatter": format_instruct}
}

# Instruction set
INSTRUCTIONS = {
    "support": [
        "Discover sources promoting",
        "Retrieve evidence confirming",
        "Locate specific arguments justifying",
        "Find statements endorsing",
        "Show points backing up"
    ],
    "attack": [
        "Discover sources dismissing",
        "Retrieve evidence disproving",
        "Locate specific arguments attacking",
        "Find statements rejecting",
        "Show points doubting"
    ]
}

# Cache registry initialisation
COMPLETED_RUNS = set()

def load_existing_cache(track_id):
    print("Scanning dataset subdirectory for existing checkpoints...")
    subdir = f"eval_{track_id}"
    csv_path = os.path.join(RESULTS_DIR, subdir, "word_ablation_results.csv")
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path).fillna("None")
            for _, row in existing_df.iterrows():
                unique_id = (
                    str(row['Model']), 
                    str(row.get('Dataset', 'Unknown')),
                    str(row['Topic']), 
                    str(row['Intent']), 
                    str(row['Instruction']), 
                    str(row['Perturbation_Type']), 
                    str(row['Removed_Text'])
                )
                COMPLETED_RUNS.add(unique_id)
        except Exception as e:
            print(f"Warning: Could not read cache from {csv_path}: {e}")

# Pre-flight filter logic
def check_needs_ablation(model_name, dataset_label, topic, intent, instruction):
    model_name, dataset_label, topic, intent, instruction = str(model_name), str(dataset_label), str(topic), str(intent), str(instruction)
    
    if (model_name, dataset_label, topic, intent, instruction, "Baseline", "None") not in COMPLETED_RUNS: 
        return True
    
    words = topic.split()
    if len(words) > 1:
        for w in words:
            if (model_name, dataset_label, topic, intent, instruction, "Query_Word", str(w)) not in COMPLETED_RUNS: 
                return True
                
    instr_words = instruction.split()
    if len(instr_words) > 1:
        for w in instr_words:
            if (model_name, dataset_label, topic, intent, instruction, "Instruction_Word", str(w)) not in COMPLETED_RUNS: 
                return True
            
    return False

# Ablation logic
def identify_target_docs(corpus, topic, intent):
    target_docs_map = {}
    for idx, doc in enumerate(corpus):
        for rel in doc.get('relations', []):
            if rel['topic'] == topic:
                if intent == "support":
                    if rel['label'] == "support": target_docs_map[idx] = "Positive"
                    elif rel['label'] == "attack": target_docs_map[idx] = "Hard_Negative"
                elif intent == "attack":
                    if rel['label'] == "attack": target_docs_map[idx] = "Positive"
                    elif rel['label'] == "support": target_docs_map[idx] = "Hard_Negative"
                break 
    return target_docs_map

def run_ablation_for_model(model_name, model, formatter, target_embeddings_dict, queries_to_run, target_docs_dict, ablation_data, dataset_label):
    batch_size = 256

    for query_info in tqdm(queries_to_run, desc=f"Ablating {model_name}"):
        topic, intent, instruction = query_info["topic"], query_info["intent"], query_info["instruction"]
        
        target_docs_map = target_docs_dict[(topic, intent, instruction)]
        if not target_docs_map: continue
        
        target_indices = list(target_docs_map.keys())
        target_embeddings = target_embeddings_dict[(topic, intent, instruction)]
        
        full_query_str = formatter(instruction, topic)
        base_q_emb = model.encode(full_query_str, convert_to_tensor=True)
        base_sims = util.cos_sim(base_q_emb, target_embeddings)[0].cpu().float().numpy()
        
        baseline_uid = (str(model_name), str(dataset_label), str(topic), str(intent), str(instruction), "Baseline", "None")
        if baseline_uid not in COMPLETED_RUNS:
            for i, doc_idx in enumerate(target_indices):
                ablation_data.append({
                    "Model": model_name, "Dataset": dataset_label, "Model_Type": "Base" if not args.adapter else "Tuned",
                    "Topic": topic, "Intent": intent, "Instruction": instruction,
                    "Doc_ID": doc_idx, "Doc_Type": target_docs_map[doc_idx], "Perturbation_Type": "Baseline",
                    "Removed_Text": "None", "Similarity": base_sims[i], "First_Order_Delta": 0.0
                })

        def process_ablation_batch(items_to_remove, perturbed_queries, pert_type):
            if not items_to_remove: return
            
            items_to_compute, queries_to_compute = [], []
            for item, p_query in zip(items_to_remove, perturbed_queries):
                unique_id = (str(model_name), str(dataset_label), str(topic), str(intent), str(instruction), str(pert_type), str(item))
                if unique_id not in COMPLETED_RUNS:
                    items_to_compute.append(item)
                    queries_to_compute.append(p_query)
            
            if not items_to_compute: return

            pert_q_embs = model.encode(queries_to_compute, convert_to_tensor=True, batch_size=batch_size)
            pert_sims_matrix = util.cos_sim(pert_q_embs, target_embeddings).cpu().float().numpy()
            
            for idx, removed_item in enumerate(items_to_compute):
                for j, doc_idx in enumerate(target_indices):
                    ablation_data.append({
                        "Model": model_name, "Dataset": dataset_label, "Model_Type": "Base" if not args.adapter else "Tuned",
                        "Topic": topic, "Intent": intent, "Instruction": instruction,
                        "Doc_ID": doc_idx, "Doc_Type": target_docs_map[doc_idx], "Perturbation_Type": pert_type,
                        "Removed_Text": removed_item, "Similarity": pert_sims_matrix[idx][j],
                        "First_Order_Delta": pert_sims_matrix[idx][j] - base_sims[j]
                    })

        words = topic.split()
        if len(words) > 1:
            process_ablation_batch(words, [formatter(instruction, " ".join(words[:i] + words[i+1:])) for i in range(len(words))], "Query_Word")

        instr_words = instruction.split()
        if len(instr_words) > 1:
            process_ablation_batch(instr_words, [formatter(" ".join(instr_words[:i] + instr_words[i+1:]), topic) for i in range(len(instr_words))], "Instruction_Word")

def load_model_helper(base_arch, adapter_path):
    config = BASE_MODELS[base_arch]
    base_path = config["path"]
    formatter = config["formatter"]
    
    print(f"Loading base model: {base_path}...")
    model = SentenceTransformer(base_path, trust_remote_code=True, model_kwargs={"torch_dtype": torch.bfloat16})
    
    if adapter_path:
        print(f"Injecting PEFT adapter from: {adapter_path}...")
        peft_model = PeftModel.from_pretrained(model[0].auto_model, adapter_path)
        model[0].auto_model = peft_model.merge_and_unload()
        
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, formatter

# Main pipeline
def run_ablation_analysis():
    load_existing_cache(args.dataset)
    
    print(f"\nInitialising execution for: {MODEL_RUN_NAME}")
    model, formatter = load_model_helper(args.base_model, args.adapter)
    
    track_id = args.dataset
    filename = f"{track_id}.json"
    dataset_label = DATASET_NAMES.get(track_id, "Unknown")
    print(f"\nStarting track: {dataset_label}")

    subdir_name = f"eval_{track_id}"
    track_results_dir = os.path.join(RESULTS_DIR, subdir_name)
    os.makedirs(track_results_dir, exist_ok=True)
    track_csv_path = os.path.join(track_results_dir, "word_ablation_results.csv")
    
    if args.local_data_dir and os.path.exists(args.local_data_dir):
        corpus_path = os.path.join(args.local_data_dir, filename)
        print(f"Loading {filename} from local directory: {args.local_data_dir}...")
    else:
        repo_name = "as9122/stance-evaluation-corpora"
        print(f"Downloading {filename} from Hugging Face: {repo_name}...")
        corpus_path = hf_hub_download(repo_id=repo_name, filename=filename, repo_type="dataset")
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    corpus = []
    topic_lookup = {t["topic_id"]: t["topic"] for t in data["topics"]}
    topics = set(t["topic"] for t in data["topics"])
    for doc in data["documents"]:
        corpus.append({
            "text": doc["text"], 
            "relations": [{"topic": topic_lookup.get(r["topic_id"], ""), "label": r["label"]} for r in doc["relations"]]
        })
        
    corpus_texts = [doc['text'] for doc in corpus]
    
    R_dict = {}
    for doc in corpus:
        for rel in doc.get('relations', []):
            key = (rel['topic'], rel['label'])
            R_dict[key] = R_dict.get(key, 0) + 1

    track_queries = []
    for topic in topics:
        for intent in ["support", "attack"]:
            R = R_dict.get((topic, intent), 0)
            if R == 0: continue 
            for instr in INSTRUCTIONS[intent]: 
                track_queries.append({"topic": topic, "intent": intent, "instruction": instr, "R": R})

    active_queries = [q for q in track_queries if check_needs_ablation(MODEL_RUN_NAME, dataset_label, q["topic"], q["intent"], q["instruction"])]

    if not active_queries:
        print(f"[*] {MODEL_RUN_NAME} is 100% complete for {dataset_label}. Bypassing.")
        return
        
    print(f"[*] Found {len(active_queries)} pending queries for {MODEL_RUN_NAME} out of {len(track_queries)} total.")
    print(f"\n[Step 1] Mapping target documents and computing corpus embeddings...")

    ablation_data = []
    target_docs_dict = {}
    target_embeddings_dict = {}
    
    corpus_embs = model.encode(corpus_texts, convert_to_tensor=True)
    
    for q_info in active_queries:
        key = (q_info["topic"], q_info["intent"], q_info["instruction"])
        docs_map = identify_target_docs(corpus, q_info["topic"], q_info["intent"])
        target_docs_dict[key] = docs_map
        
        target_indices = list(docs_map.keys())
        target_embeddings_dict[key] = corpus_embs[target_indices]

    print(f"\n[Step 2] Executing ablations for {MODEL_RUN_NAME}...")
    run_ablation_for_model(MODEL_RUN_NAME, model, formatter, target_embeddings_dict, active_queries, target_docs_dict, ablation_data, dataset_label)
    
    del corpus_embs, target_embeddings_dict
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    print(f"\n[*] Saving incremental ablation data for {MODEL_RUN_NAME} to {track_csv_path}...")
    if ablation_data:
        cols = ['Model', 'Dataset', 'Model_Type', 'Topic', 'Intent', 'Instruction', 'Doc_ID', 'Doc_Type', 'Perturbation_Type', 'Removed_Text', 'Similarity', 'First_Order_Delta']
        new_df = pd.DataFrame(ablation_data)[cols]
        if os.path.exists(track_csv_path):
            new_df.to_csv(track_csv_path, mode='a', header=False, index=False)
            print(f"[*] Appended {len(new_df)} new rows to existing checkpoint.")
        else:
            new_df.to_csv(track_csv_path, index=False)
            print(f"[*] Created new file with {len(new_df)} rows.")
    else:
        print("[*] No new ablations generated.")

    for item in ablation_data:
        COMPLETED_RUNS.add((str(item['Model']), str(item['Dataset']), str(item['Topic']), str(item['Intent']), str(item['Instruction']), str(item['Perturbation_Type']), str(item['Removed_Text'])))

    print("\nEvaluation track complete!")

if __name__ == "__main__":
    run_ablation_analysis()