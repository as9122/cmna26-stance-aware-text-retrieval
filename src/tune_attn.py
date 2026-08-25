import os
import json
import torch
import gc
import argparse
from datasets import Dataset, load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses
)
from peft import LoraConfig
from torch.utils.data import Sampler
from collections import defaultdict
import random

# Command line arguments
parser = argparse.ArgumentParser(description="Run Data-Centric Stance Fine-Tuning")
parser.add_argument("--model_name", type=str, default="Qwen3-Embedding-8B", choices=["BGE-Large", "Instructor-XL", "Qwen3-Embedding-8B"])
parser.add_argument("--rank", type=int, default=None, help="LoRA rank")
parser.add_argument("--strategy", type=str, default="Homogeneous", choices=["Homogeneous", "Mixed", "Mixed_Aug"], help="Which data-centric intervention to train on.")
parser.add_argument("--local_data_path", type=str, default=None, help="Optional local path to a JSON dataset. If provided, overrides Hugging Face download.")
parser.add_argument("--run_id_suffix", type=str, default="", help="Optional custom suffix for the run identifier.")

args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "results", "finetuned_models")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def format_instruct(instruction_text, topic_text):
    return f"Instruct: {instruction_text}\nQuery: {topic_text}"

def format_instructor(instruction_text, topic_text):
    return f"{instruction_text} {topic_text}"

MODELS_CONFIG = {
    "BGE-Large": {
        "path": "BAAI/bge-large-en-v1.5",
        "formatter": format_instruct,
        "target_modules": ["query", "key", "value", "dense"]
    },
    "Instructor-XL": {
        "path": "hkunlp/instructor-xl",
        "formatter": format_instructor,
        "target_modules": ["q", "k", "v", "o"]
    },
    "Qwen3-Embedding-8B": {
        "path": "Qwen/Qwen3-Embedding-8B",
        "formatter": format_instruct,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    }
}

class UniqueClaimBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        claim_to_indices = defaultdict(list)
        for i, claim in enumerate(self.dataset["claim"]):
            claim_to_indices[claim].append(i)

        for indices in claim_to_indices.values():
            random.shuffle(indices)

        available_claims = list(claim_to_indices.keys())

        while available_claims:
            random.shuffle(available_claims)
            selected_claims = available_claims[:self.batch_size]

            batch = []
            for claim in selected_claims:
                idx = claim_to_indices[claim].pop()
                batch.append(idx)
                
                if not claim_to_indices[claim]:
                    available_claims.remove(claim)

            if batch:
                yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

# Dataset Preparation
def load_and_format_dataset(model_name, strategy, local_path):
    if local_path and os.path.exists(local_path):
        print(f"Loading dataset from local path: {local_path}...")
        with open(local_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    else:
        file_name = f"{strategy.lower()}.json"
        repo_name = "as9122/stance-tfu-training"
        print(f"Loading {file_name} from Hugging Face: {repo_name}...")
        
        # Download via HF and convert to a list of dicts to match standard JSON formatting
        hf_dataset = load_dataset(repo_name, data_files=file_name, split="train")
        raw_data = [item for item in hf_dataset]
        
    formatter = MODELS_CONFIG[model_name]["formatter"]
    
    formatted_data = {"anchor": [], "positive": [], "negative": [], "claim": []}
    
    for item in raw_data:
        anchor_text = formatter(item["instruction"], item["claim"])
        formatted_data["anchor"].append(anchor_text)
        formatted_data["positive"].append(item["positive"])
        formatted_data["negative"].append(item["negative"])
        formatted_data["claim"].append(item["claim"])
            
    return Dataset.from_dict(formatted_data)

# Custom Trainer for Unique Claim Batching
class UniqueClaimTrainer(SentenceTransformerTrainer):
    def get_train_dataloader(self):
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        
        sampler = UniqueClaimBatchSampler(train_dataset, self.args.train_batch_size)
        
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=sampler,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

def get_model(model_name, current_rank=None):
    config = MODELS_CONFIG[model_name]
    model_kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": True, "attn_implementation": "eager"}
    
    if current_rank is not None:
        print(f"\nInitialising {model_name} with LoRA Rank {current_rank}...")
        lora_config = LoraConfig(
            r=current_rank, 
            lora_alpha=current_rank * 2, 
            target_modules=config["target_modules"], 
            lora_dropout=0.05, 
            bias="none", 
            task_type="FEATURE_EXTRACTION"
        )
        model = SentenceTransformer(config["path"], model_kwargs=model_kwargs, trust_remote_code=True)
        model.add_adapter(lora_config)
    else:
        print(f"\nInitialising {model_name} for Full Fine-Tuning...")
        model = SentenceTransformer(config["path"], model_kwargs=model_kwargs, trust_remote_code=True)
        
    return model

def run_training(model_name, strategy, current_rank=None, local_path=None):
    run_identifier = f"{model_name}_{strategy}"
    if args.run_id_suffix:
        run_identifier += f"_{args.run_id_suffix}"
    
    final_save_path = os.path.join(OUTPUT_DIR, f"{run_identifier}_final")
    if os.path.exists(final_save_path):
        print(f"\nSkipping {run_identifier}... Final model exists at {final_save_path}")
        return

    dataset = load_and_format_dataset(model_name, strategy, local_path)
    model = get_model(model_name, current_rank)
    standard_loss_fn = losses.MultipleNegativesRankingLoss(model)
    
    is_t5 = "Instructor" in model_name
    use_bfloat16 = (current_rank is not None) or is_t5
    
    bs = 8
    grad_acc = 2
    
    training_args = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, f"{run_identifier}_checkpoints"),
        num_train_epochs=2,                 
        per_device_train_batch_size=bs,      
        gradient_accumulation_steps=grad_acc,      
        gradient_checkpointing=True,   
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=not use_bfloat16,
        bf16=use_bfloat16,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        remove_unused_columns=False, 
    )

    trainer = UniqueClaimTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=standard_loss_fn,
    )
    
    print(f"\nStarting training for {run_identifier}...")
    trainer.train()
    model.save_pretrained(final_save_path)
    print(f"\nTraining complete. Model saved to {final_save_path}")

    del model, trainer, standard_loss_fn, dataset
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

if __name__ == "__main__":
    run_training(model_name=args.model_name, strategy=args.strategy, current_rank=args.rank, local_path=args.local_data_path)