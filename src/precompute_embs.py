import os
import gc
import json
import torch
import argparse
from tqdm import tqdm
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
import nltk

# NLTK Setup
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

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
parser = argparse.ArgumentParser(description="Precompute Embeddings for Evaluation Tracks")
parser.add_argument("--base_model", type=str, required=True, choices=["BGE-Large", "Instructor-XL", "Qwen3-Embedding-8B"], 
                    help="The base foundation model architecture to use.")
parser.add_argument("--adapter", type=str, default=None, 
                    help="Optional: Hugging Face Hub ID or a local directory path to a trained LoRA adapter.")
parser.add_argument("--datasets", nargs="+", choices=DATASET_KEYS, default=DATASET_KEYS, 
                    help="Which datasets to process. If omitted, runs all available tracks.")
parser.add_argument("--local_data_dir", type=str, default=None, 
                    help="Optional: Local directory path containing the evaluation JSON corpora. If omitted, downloads from HF.")
parser.add_argument("--run_name", type=str, default=None, 
                    help="Optional: Identifier for this run (e.g., 'BGE-Large_Mixed'). If omitted, an automatic name is generated.")
args = parser.parse_args()

# Determine the model display name
if args.run_name:
    MODEL_RUN_NAME = args.run_name
else:
    MODEL_RUN_NAME = args.base_model if not args.adapter else f"{args.base_model}_{os.path.basename(args.adapter.strip('/'))}"

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
PRECOMPUTED_DIR = os.path.join(RESULTS_DIR, "precomputed_vectors")
os.makedirs(PRECOMPUTED_DIR, exist_ok=True)

BATCH_SIZE = 64

# Instructions mapping
INSTRUCTIONS = {
    "Seen": {
        "support": [
            "Find evidence backing", "Retrieve arguments in favor of",
            "Search for statements validating", "Highlight information corroborating",
            "Uncover points upholding", "Show claims affirming",
            "Provide reasoning that supports", "Identify excerpts substantiating",
            "Extract statements advocating for", "Gather facts verifying"
        ],
        "attack": [
            "Find evidence refuting", "Retrieve arguments against",
            "Search for statements invalidating", "Highlight information contradicting",
            "Uncover points opposing", "Show claims challenging",
            "Provide reasoning that undermines", "Identify excerpts disputing",
            "Extract statements arguing against", "Gather facts debunking"
        ]
    },
    "Unseen": {
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
}

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

# Main Execution
def main():
    print(f"\nInitializing precomputation for: {MODEL_RUN_NAME}")
    model, formatter = load_model_helper(args.base_model, args.adapter)
    
    for track_id in args.datasets:
        filename = f"{track_id}.json"
        out_path = os.path.join(PRECOMPUTED_DIR, f"{track_id}_{MODEL_RUN_NAME}.pt")
        
        if os.path.exists(out_path): 
            print(f"\nSkipping {MODEL_RUN_NAME} on {track_id} (Already precomputed at {out_path})")
            continue

        print(f"\n{'*'*50}\nSTARTING TRACK: {track_id.upper()}\n{'*'*50}")
        
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
        
        # 1. Build Corpus Metadata & Tokenize for BM25
        corpus_metadata = []
        for doc in tqdm(data["documents"], desc="Tokenizing Corpus"):
            hydrated_relations = [{"topic": topic_lookup.get(r["topic_id"], ""), "label": r["label"]} for r in doc["relations"]]
            corpus_metadata.append({
                "text": doc["text"],
                "relations": hydrated_relations,
                "lexical_tokens": [w.lower() for w in nltk.word_tokenize(doc["text"]) if w.isalpha()]
            })

        # 2. Build Query Metadata & Tokenize
        unique_topics = list(topic_lookup.values())
        query_metadata = []
        for topic in unique_topics:
            for instr_type, intents in INSTRUCTIONS.items():
                for intent, instructions in intents.items():
                    for instr in instructions:
                        query_metadata.append({
                            "formatted_query": formatter(instr, topic),
                            "topic": topic,
                            "intent": intent,
                            "instruction": instr,
                            "instruction_type": instr_type,
                            "lexical_tokens": [w.lower() for w in nltk.word_tokenize(topic) if w.isalpha()]
                        })

        # Strictly use plain string inputs for all models to match training behavior
        model_corpus_input = [d["text"] for d in corpus_metadata]
        model_query_input = [q["formatted_query"] for q in query_metadata]

        # 3. Compute and Save Embeddings
        print("Encoding Vectors...")
        with torch.no_grad():
            corpus_embs = model.encode(model_corpus_input, convert_to_tensor=True, batch_size=BATCH_SIZE, show_progress_bar=True)
            query_embs = model.encode(model_query_input, convert_to_tensor=True, batch_size=BATCH_SIZE, show_progress_bar=True)
        
        package = {
            "dataset": track_id,
            "model_name": MODEL_RUN_NAME,
            "corpus_metadata": corpus_metadata,
            "query_metadata": query_metadata,
            "corpus_embs": corpus_embs.cpu(),
            "query_embs": query_embs.cpu()
        }
        
        torch.save(package, out_path)
        print(f"Saved precomputed package to: {out_path}")

        # Clean tensors for next track
        del corpus_embs, query_embs
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    print("\nPrecomputation complete!")

if __name__ == "__main__":
    main()