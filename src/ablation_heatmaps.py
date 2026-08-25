import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm
import re
import textwrap
import random
from huggingface_hub import hf_hub_download

# Command line arguments
parser = argparse.ArgumentParser(description="Generate word ablation heatmaps")
parser.add_argument("--csv_path", type=str, required=True, 
                    help="Path to the word_ablation_results.csv file")
parser.add_argument("--output_dir", type=str, default="./ablation_figs", 
                    help="Directory to save the generated heatmaps")
parser.add_argument("--base_model_name", type=str, required=True, 
                    help="Exact name of the base model as it appears in the CSV (e.g., 'Qwen3-Embedding-8B')")
parser.add_argument("--tuned_model_name", type=str, required=True, 
                    help="Exact name of the tuned model as it appears in the CSV (e.g., 'Qwen3-Embedding-8B_Mixed')")
parser.add_argument("--include_doc_text", action="store_true", default=False,
                    help="If set, displays the full document argument text on the y-axis. Defaults to tags and Doc IDs only.")
parser.add_argument("--max_plots", type=int, default=10, 
                    help="Maximum number of topic/instruction pairs to plot. Randomly samples if exceeding.")
parser.add_argument("--seed", type=int, default=42, 
                    help="Random seed for sampling to ensure reproducible plot selections.")
parser.add_argument("--allowed_words", type=str, nargs="+", default=None, 
                    help="Optional list of specific ablated words to include on the x-axis. Omitting shows all.")
parser.add_argument("--allowed_doc_types", type=str, nargs="+", default=None, 
                    help="Optional list of document types to include (e.g., Positive Hard_Negative).")
parser.add_argument("--max_docs_per_type", type=int, default=None, 
                    help="Maximum number of documents to show per document type.")
parser.add_argument("--v_min", type=float, default=-0.20, help="Minimum value for the color scale.")
parser.add_argument("--v_max", type=float, default=0.20, help="Maximum value for the color scale.")
parser.add_argument("--wrap_width", type=int, default=55, help="Character wrap width for document text.")
args = parser.parse_args()

# Configuration and mapping
FILENAME_LIMIT = 60
X_LABEL_WRAP_WIDTH = 15 
FONT_SIZE = 10    

EVAL_TRACKS = {
    "averitec": "averitec.json",
    "tfu_evaluation": "tfu_evaluation.json",
    "tfu_training_heldout": "tfu_training_heldout.json",
    "argtumour": "argtumour.json",
    "tfu_validation_gpt5": "tfu_validation_gpt5.json",
    "tfu_validation_qwen": "tfu_validation_qwen.json"
}

DATASET_NAMES = {
    "averitec": "AVeriTeC Arguments",
    "tfu_evaluation": "TFU Evaluation Arguments",
    "tfu_training_heldout": "TFU Training Arguments (20% held-out)",
    "argtumour": "ArgTumour Arguments",
    "tfu_validation_gpt5": "TFU Validation Arguments (GPT-5)",
    "tfu_validation_qwen": "TFU Validation Arguments (Qwen3-8B)"
}

def safe_filename(text):
    clean = re.sub(r'[^a-zA-Z0-9\s-]', '', str(text))
    clean = clean.strip().replace(' ', '_')
    return clean[:FILENAME_LIMIT]

def format_doc_text_only(doc_id, dataset_label, corpus_maps):
    try:
        doc_id = int(doc_id)
    except ValueError:
        pass 
    
    full_text = corpus_maps.get(dataset_label, {}).get(doc_id, "Text Not Found")
    return "\n".join(textwrap.wrap(full_text, width=args.wrap_width))

def get_tag_info(tag_type):
    if tag_type == "Positive": return "[POS]", "darkgreen"
    if tag_type == "Hard_Negative": return "[NEG]", "darkred"
    return "[?]", "black"

def run_visualisation():
    if not os.path.exists(args.csv_path):
        print(f"Error: Could not find CSV file at {args.csv_path}")
        return

    corpus_maps = {}
    if args.include_doc_text:
        print("Downloading and loading corpora from Hugging Face...")
        repo_name = "as9122/stance-evaluation-corpora"
        
        for track_id, filename in EVAL_TRACKS.items():
            dataset_label = DATASET_NAMES.get(track_id, "Unknown")
            try:
                corpus_path = hf_hub_download(repo_id=repo_name, filename=filename, repo_type="dataset")
                with open(corpus_path, 'r', encoding='utf-8') as f:
                    corpus_data = json.load(f)
                    
                docs_list = corpus_data.get("documents", [])
                corpus_maps[dataset_label] = {i: doc.get("text", "") for i, doc in enumerate(docs_list)}
            except Exception as e:
                print(f"  - Warning: Failed to load {filename}. Error: {e}")

    print("Loading dataframe...")
    df = pd.read_csv(args.csv_path)
    
    df = df[df['Perturbation_Type'] != 'Baseline'].copy()
    df['Removed_Text'] = df['Removed_Text'].astype(str)
    df.loc[df['Removed_Text'] == 'nan', 'Removed_Text'] = 'None'
    df['Col_Key'] = df['Perturbation_Type'] + "::" + df['Removed_Text']

    # Extract unique topic and instruction pairs directly from the CSV
    target_pairs = df[['Topic', 'Instruction']].drop_duplicates().to_dict('records')
    
    # Randomly sample if the number of unique pairs exceeds max_plots
    if len(target_pairs) > args.max_plots:
        print(f"Found {len(target_pairs)} unique scenarios. Randomly sampling {args.max_plots} (Seed: {args.seed})...")
        random.seed(args.seed)
        sampled_pairs = random.sample(target_pairs, args.max_plots)
    else:
        print(f"Found {len(target_pairs)} unique target scenarios to plot.")
        sampled_pairs = target_pairs

    os.makedirs(args.output_dir, exist_ok=True)
    
    for query_info in tqdm(sampled_pairs, desc="Generating Heatmaps"):
        topic = query_info['Topic']
        instr = query_info['Instruction']
        
        subset = df[(df['Topic'] == topic) & (df['Instruction'] == instr)].copy()
        
        if subset.empty:
            continue
            
        base_subset = subset[subset['Model'] == args.base_model_name]
        tuned_subset = subset[subset['Model'] == args.tuned_model_name]
        
        if base_subset.empty or tuned_subset.empty:
            print(f"  - Warning: Missing Base or Tuned data for Topic: '{topic}'. Skipping.")
            continue 
            
        # Build full x-axis
        instr_words = str(instr).split()
        claim_words = str(topic).split()
        
        chunk_df = subset[subset['Perturbation_Type'] == 'Query_Chunk']
        chunks = [c for c in pd.unique(chunk_df['Removed_Text']) if c != 'None']
        has_chunks = len(chunks) > 0
        
        gap_col = "   "
        x_axis_map = [] 
        for w in instr_words: x_axis_map.append((w, f"Instruction_Word::{w}", "Instruction"))
        x_axis_map.append((gap_col, "GAP1", "GAP"))
        for w in claim_words: x_axis_map.append((w, f"Query_Word::{w}", "Claim"))
        
        if has_chunks:
            x_axis_map.append((gap_col, "GAP2", "GAP"))
            for c in chunks: x_axis_map.append((c, f"Query_Chunk::{c}", "Chunk"))

        # Apply word (x-axis) filtering
        if args.allowed_words:
            x_axis_map = [x for x in x_axis_map if x[2] == "GAP" or x[0] in args.allowed_words]
            x_axis_map = [x for x in x_axis_map if not (x[2] == "GAP" and x_axis_map.index(x) in [0, len(x_axis_map)-1])]
                
        # Determine full row list (docs)
        base_temp_matrix = base_subset.pivot_table(index='Doc_ID', columns='Col_Key', values='First_Order_Delta', aggfunc='mean')
        tuned_temp_matrix = tuned_subset.pivot_table(index='Doc_ID', columns='Col_Key', values='First_Order_Delta', aggfunc='mean')
        
        all_docs = list(set(base_temp_matrix.index.dropna().tolist() + tuned_temp_matrix.index.dropna().tolist()))
        id_to_type = dict(zip(subset['Doc_ID'], subset['Doc_Type']))
        
        type_priority = {
            "Positive": 0, "Hard_Negative": 1
        }
        all_docs.sort(key=lambda x: (type_priority.get(id_to_type.get(x), 5), x))

        # Apply document (y-axis) filtering
        if args.allowed_doc_types:
            all_docs = [d for d in all_docs if id_to_type.get(d) in args.allowed_doc_types]

        if args.max_docs_per_type is not None:
            filtered_docs = []
            type_counts = {}
            for d in all_docs:
                doc_t = id_to_type.get(d)
                type_counts[doc_t] = type_counts.get(doc_t, 0) + 1
                if type_counts[doc_t] <= args.max_docs_per_type:
                    filtered_docs.append(d)
            all_docs = filtered_docs
            
        full_keys = [k for _, k, _ in x_axis_map]

        if not all_docs or not full_keys:
            continue

        # Rebuild matrices with cropped lists
        base_matrix = base_temp_matrix.reindex(index=all_docs, columns=full_keys)
        tuned_matrix = tuned_temp_matrix.reindex(index=all_docs, columns=full_keys)
        
        if "GAP1" in full_keys:
            base_matrix["GAP1"] = np.nan; tuned_matrix["GAP1"] = np.nan
        if "GAP2" in full_keys:
            base_matrix["GAP2"] = np.nan; tuned_matrix["GAP2"] = np.nan

        # Prepare labels
        dataset_label = subset['Dataset'].iloc[0] if 'Dataset' in subset.columns else "Unknown"
        
        y_labels_text = []
        for did in all_docs:
            tag_type = id_to_type.get(did)
            t_text, _ = get_tag_info(tag_type)
            if args.include_doc_text:
                doc_text = format_doc_text_only(did, dataset_label, corpus_maps)
                y_labels_text.append(f"{t_text} {doc_text}")
            else:
                y_labels_text.append(f"{t_text} Doc {did}")
            
        y_tags_types = [id_to_type.get(did) for did in all_docs]
        x_labels_text = [label for label, _, _ in x_axis_map]
        x_labels_wrapped = [textwrap.fill(l, X_LABEL_WRAP_WIDTH) if l != gap_col else "" for l in x_labels_text]

        # Plotting
        total_lines = sum([l.count('\n') + 1 for l in y_labels_text])
        dynamic_height = max(5, total_lines * (0.6 if args.include_doc_text else 0.4))
        dynamic_width = max(8, len(x_axis_map) * 0.7) 
        
        fig, axes = plt.subplots(1, 3, figsize=(dynamic_width * 2, dynamic_height), gridspec_kw={'width_ratios': [1, 1, 0.05]})
        axes[1].sharey(axes[0])
        fig.patch.set_facecolor('white')
        
        for ax, matrix, m_name in zip(axes[:2], [base_matrix, tuned_matrix], [args.base_model_name, args.tuned_model_name]):
            ax.set_facecolor('white') 
            mask = matrix.isnull()

            sns.heatmap(
                matrix, cmap='coolwarm_r', center=0, vmin=args.v_min, vmax=args.v_max,
                annot=True, fmt=".2f", linewidths=0.5, linecolor='black',
                xticklabels=x_labels_wrapped, yticklabels=y_labels_text if ax == axes[0] else False,
                mask=mask, ax=ax, cbar=(ax == axes[1]), 
                cbar_ax=axes[2] if ax == axes[1] else None,
                cbar_kws={'label': 'First Order Delta'} if ax == axes[1] else None
            )
            
            ax.set_title(f"{'Base' if ax == axes[0] else 'Tuned'} Model: {m_name}", fontsize=FONT_SIZE+2, pad=35)
            ax.set_xlabel("", fontsize=FONT_SIZE + 1)
            ax.set_ylabel("Documents" if ax == axes[0] else "", fontsize=FONT_SIZE + 1)
            ax.tick_params(axis='x', rotation=45, labelsize=FONT_SIZE)

            # Dynamic section headings
            sections = {'Instruction': [], 'Claim': [], 'Chunk': []}
            for i, (_, _, sec_type) in enumerate(x_axis_map):
                if sec_type in sections:
                    sections[sec_type].append(i)

            for sec_name, indices in sections.items():
                if indices:
                    start_idx = indices[0]
                    end_idx = indices[-1]
                    center_idx = start_idx + (end_idx - start_idx) / 2
                    
                    ax.text(center_idx + 0.5, 1.01, f"{sec_name} Word Ablation", 
                            ha='center', va='bottom', transform=ax.get_xaxis_transform(),
                            fontweight='bold', fontsize=FONT_SIZE, color='black', 
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))

        # Manual tags formatting
        axes[0].set_yticks(np.arange(len(y_labels_text)) + 0.5)
        axes[0].set_yticklabels(y_labels_text, rotation=0, ha='right', fontsize=9)
        
        for tick_label, tag_type in zip(axes[0].get_yticklabels(), y_tags_types):
            _, t_color = get_tag_info(tag_type)
            tick_label.set_color(t_color)
            tick_label.set_fontweight('bold')

        full_title = f"First-Order Ablation Delta\nClaim: {topic}\nInstruction: {instr}"
        plt.suptitle(full_title, fontsize=FONT_SIZE + 4, fontweight='bold', y=1.05) 
        
        plt.tight_layout()
        
        safe_topic = safe_filename(topic)[:30]
        safe_instr = safe_filename(instr)[:30]
        filename = f"ablation_{safe_topic}_{safe_instr}.png"
        
        plt.savefig(os.path.join(args.output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\nAll plots saved to: {args.output_dir}")

if __name__ == "__main__":
    run_visualisation()