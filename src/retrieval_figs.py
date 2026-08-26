import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Configuration 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")

parser = argparse.ArgumentParser(description="Generate retrieval figures from experiment results.")
parser.add_argument(
    "--lambdas", 
    nargs="*", 
    type=float,
    default=[], 
    help="Optional: List of lambda weights to append to the plots (e.g., --lambdas 0.1 0.15)."
)
parser.add_argument(
    "--hybrid_only", 
    action="store_true", 
    help="If set, only plots the hybrid runs and hides the base models. Requires --lambdas."
)
args = parser.parse_args()

# Validation check
if args.hybrid_only and not args.lambdas:
    print("Error: You must provide at least one lambda value via --lambdas when using --hybrid_only.")
    exit(1)

# 1. Define the models you want to plot (in the exact order you want them)
BASE_MODELS_TO_PLOT = [
    "BGE-Large_base",
    "BGE-Large_homogeneous",
    "BGE-Large_mixed",
    "BGE-Large_mixed_aug",
    # "Instructor-XL_base",
    # "Instructor-XL_homogeneous",
    # "Instructor-XL_mixed",
    # "Instructor-XL_mixed_aug",
    # "Qwen3-Embedding-8B_base",
    # "Qwen3-Embedding-8B_homogeneous",
    # "Qwen3-Embedding-8B_mixed",
    # "Qwen3-Embedding-8B_mixed_aug"
]

# 2. Define how they should look on the y-axis
MODEL_DISPLAY_NAMES = {
    "BGE-Large_base": "BGE Base",
    "BGE-Large_homogeneous": "BGE Homogeneous",
    "BGE-Large_mixed": "BGE Mixed",
    "BGE-Large_mixed_aug": "BGE Mixed + Aug",
    "Instructor-XL_base": "Instructor Base",
    "Instructor-XL_homogeneous": "Instructor Homogeneous",
    "Instructor-XL_mixed": "Instructor Mixed",
    "Instructor-XL_mixed_aug": "Instructor Mixed + Aug",
    "Qwen3-Embedding-8B_base": "Qwen3 Base",
    "Qwen3-Embedding-8B_homogeneous": "Qwen3 Homogeneous",
    "Qwen3-Embedding-8B_mixed": "Qwen3 Mixed",
    "Qwen3-Embedding-8B_mixed_aug": "Qwen3 Mixed + Aug"
}

# 3. Define the datasets you want to plot (in the exact order you want them)
EXPECTED_DATASET_ORDER = [
    "eval_tfu_training_heldout",
    "eval_tfu_validation_gpt5",
    "eval_tfu_validation_qwen",
    "eval_argtumour",
    "eval_tfu_evaluation",
    "eval_averitec",
]

# 4. Clean display names for datasets
DATASET_DISPLAY_NAMES = {
    "eval_tfu_training_heldout": "TFU Training (20% held-out)",
    "eval_tfu_validation_gpt5": "TFU Validation (GPT-5)",
    "eval_tfu_validation_qwen": "TFU Validation (Qwen3-8B)",
    "eval_argtumour": "ArgTumour",
    "eval_tfu_evaluation": "TFU Evaluation",
    "eval_averitec": "AVeriTeC",
} 

# Standard styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

def plot_ndcg(df, models_to_plot, output_dir):
    """Plots Average NDCG for Support, Attack, and Overall across datasets."""
    if 'NDCG@10' not in df.columns:
        return

    # Filter and order datasets based on EXPECTED_DATASET_ORDER
    available_datasets = df['Dataset'].unique()
    datasets = [d for d in EXPECTED_DATASET_ORDER if d in available_datasets]
    
    intents_to_plot = list(df['Intent'].unique()) + ["overall"]

    for intent in intents_to_plot:
        subset = df if intent == "overall" else df[df['Intent'] == intent]
        n_plots = len(datasets)
        
        subplot_height = max(2.0, len(models_to_plot) * 0.4)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, subplot_height * n_plots), sharex=True)
        if n_plots == 1: 
            axes = [axes]
        
        for plot_idx, dataset_name in enumerate(datasets):
            ax = axes[plot_idx]
            d_subset = subset[subset['Dataset'] == dataset_name]
            
            # Reindex enforces the exact ordering specified by the user
            avg_data = d_subset.groupby('Model')[['NDCG@10']].mean().reindex(models_to_plot).fillna(0)
            avg_data.index = [MODEL_DISPLAY_NAMES.get(m, m) for m in avg_data.index]
            
            avg_data.plot(kind='barh', ax=ax, color='#3498db', legend=False, width=0.7)
            ax.invert_yaxis()
            
            # Add separator lines between model families
            for j in range(1, len(models_to_plot)):
                if models_to_plot[j-1].split('-')[0] != models_to_plot[j].split('-')[0]:
                    ax.axhline(j - 0.5, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
            
            ax.set_title(f"Dataset: {DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)}", fontsize=12, loc='left', fontweight='bold')
            ax.set_xlim(0, 1.0)
            ax.set_ylabel("")
            
            # Add bold white text inside the bars
            for c in ax.containers:
                bar_labels = [f'{v.get_width():.3f}' if v.get_width() > 0.01 else '' for v in c]
                ax.bar_label(c, labels=bar_labels, label_type='edge', padding=3, color='black', fontsize=9)
                
        title_intent = "Overall" if intent == "overall" else f"{intent.capitalize()} Intent"
        
        plt.tight_layout()
        # Push the title completely outside the plot area to prevent overlap
        plt.suptitle(f"Average NDCG@10: {title_intent}", y=1.05, fontsize=16)
        
        os.makedirs(output_dir, exist_ok=True)
        # bbox_inches='tight' will automatically expand the save area to capture the title
        plt.savefig(os.path.join(output_dir, f"NDCG_{intent}.png"), bbox_inches='tight')
        plt.close()

def plot_precision_stacked(df, models_to_plot, output_dir):
    """Plots Precision stacked failures for Support, Attack, and Overall across datasets."""
    if 'Precision@R' not in df.columns:
        return

    # Filter and order datasets based on EXPECTED_DATASET_ORDER
    available_datasets = df['Dataset'].unique()
    datasets = [d for d in EXPECTED_DATASET_ORDER if d in available_datasets]
    
    intents_to_plot = list(df['Intent'].unique()) + ["overall"]
    
    colors = ["#10AD10", "#D41010", '#bdc3c7', '#7f8c8d']
    labels = ['Positive', 'Hard Negative', 'Semi-Hard Negative', 'Easy Negative']
    
    # Custom text colors: Make the 3rd bar (Semi-Hard Negative) use dark grey text for readability
    text_colors = ["white", "white", "#333333", "white"]
    
    legend_handles = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    
    for intent in intents_to_plot:
        subset = df if intent == "overall" else df[df['Intent'] == intent]
        n_plots = len(datasets)
        
        subplot_height = max(2.0, len(models_to_plot) * 0.4)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, subplot_height * n_plots), sharex=True)
        if n_plots == 1: 
            axes = [axes]
        
        for plot_idx, dataset_name in enumerate(datasets):
            ax = axes[plot_idx]
            d_subset = subset[subset['Dataset'] == dataset_name]
            
            avg_data = d_subset.groupby('Model')[['Precision@R', 'Stance_Error_R', 'Irrelevant_Correct_Stance_R', 'Irrelevant_Incorrect_Stance_R']].mean().reindex(models_to_plot).fillna(0)
            avg_data.index = [MODEL_DISPLAY_NAMES.get(m, m) for m in avg_data.index]
            
            avg_data.plot(kind='barh', stacked=True, ax=ax, color=colors, legend=False, width=0.7)
            ax.invert_yaxis()
            
            # Add separator lines between model families
            for j in range(1, len(models_to_plot)):
                if models_to_plot[j-1].split('-')[0] != models_to_plot[j].split('-')[0]:
                    ax.axhline(j - 0.5, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
            
            ax.set_title(f"Dataset: {DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)}", fontsize=12, loc='left', fontweight='bold')
            ax.set_xlim(0, 1.0)
            ax.set_ylabel("")
            
            # Add bold text inside the bars using the custom text_colors array
            for i, c in enumerate(ax.containers):
                bar_labels = [f'{v.get_width():.3f}' if v.get_width() > 0.05 else '' for v in c]
                ax.bar_label(c, labels=bar_labels, label_type='center', color=text_colors[i % 4], fontsize=9, fontweight='bold')
                
        title_intent = "Overall" if intent == "overall" else f"{intent.capitalize()} Intent"
        
        plt.tight_layout()
        # Anchor the legend to the very top edge, and push the title even higher above it
        fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=10)
        plt.suptitle(f"Precision@R Breakdown: {title_intent}", y=1.12, fontsize=16)
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"Precision_Stacked_{intent}.png"), bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    plots_dir = os.path.join(RESULTS_DIR, "plots")
    
    # Build the models list so hybrids group directly under their base model
    models_to_plot = []
    for model in BASE_MODELS_TO_PLOT:
        # Add the base model unless hybrid_only is flagged
        if not args.hybrid_only:
            models_to_plot.append(model)
            
        # Add the corresponding hybrid variants right after
        for lam in args.lambdas:
            hybrid_name = f"{model}_Hybrid_RSF_{lam}"
            models_to_plot.append(hybrid_name)
            
            # Map the hybrid string to a clean display name using the base model's display name
            base_disp = MODEL_DISPLAY_NAMES.get(model, model)
            MODEL_DISPLAY_NAMES[hybrid_name] = f"{base_disp} (λ={lam})"
    
    all_dfs = []
    print("Scanning for evaluation results...")
    for folder in os.listdir(RESULTS_DIR):
        if not folder.startswith("eval_"):
            continue
            
        csv_path = os.path.join(RESULTS_DIR, folder, "experiment_results.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df['Dataset'] = folder
                all_dfs.append(df)
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")
                
    if not all_dfs:
        print("No experiment_results.csv files found. Exiting.")
    else:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Filter to unseen instructions if present
        if 'Instruction_Type' in combined_df.columns:
            combined_df = combined_df[combined_df['Instruction_Type'].str.lower() == "unseen"]

        # Filter dataframe strictly to requested models
        combined_df = combined_df[combined_df['Model'].isin(models_to_plot)]

        print(f"Plotting {len(models_to_plot)} models across {combined_df['Dataset'].nunique()} datasets...")
        plot_ndcg(combined_df, models_to_plot, plots_dir)
        plot_precision_stacked(combined_df, models_to_plot, plots_dir)
        
        print(f"All figures saved to {plots_dir}")