import os
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration
PLOT_BG_COLOR = "#FFFFFF"
GLOBAL_TEXT_COLOR = "#000000"
FONT_SIZE = 12

STANCE_KEYWORDS = {
    "benefits", "good", "agree", "positive", "downsides", "bad", "disagree", "negative",
    "backing", "favor", "validating", "corroborating", "upholding", "affirming", 
    "supports", "substantiating", "advocating", "verifying", "promoting", 
    "confirming", "justifying", "proving", "agreeing", "endorsing",
    "refuting", "against", "invalidating", "contradicting", "opposing", "challenging", 
    "undermines", "disputing", "arguing", "debunking", "dismissing", "disproving", 
    "attacking", "rejecting", "doubting", "criticisms"
}

VERSION_DISPLAY = {
    "Base": "Base",
    "Homogeneous": "Homogeneous",
    "Mixed": "Mixed",
    "Mixed_Aug": "Mixed + Augmentation"
}

# Ensure these exactly match what ablation.py produces
DATASET_DISPLAY = {
    "eval_general_test": "TFU Training Arguments (20% held-out)",
    "eval_validation": "TFU Validation Arguments (GPT-5)",
    "eval_validation_qwen": "TFU Validation Arguments (Qwen3-8B)",
    "eval_medical": "ArgTumour Arguments",
    "eval_evaluation": "TFU Evaluation Arguments",
    "eval_averitec": "AVeriTeC Arguments",
}

DOC_DISPLAY = {
    "Relevant": "Relevant",
    "Positive": "Positive",
    "Hard_Negative": "Hard Negative"
}

# Plotting Functions
def generate_rs_metrics(df, focus, output_dir):
    metric_abbr = f"R{focus[0]}S"
    print(f"Generating Relative {focus} Sensitivity ({metric_abbr}) Plots...")

    word_df = df[df['Perturbation_Type'].isin(['Instruction_Word', 'Query_Word'])].copy()
    word_df['Abs_Delta'] = word_df['First_Order_Delta'].abs()

    if focus == 'Instruction':
        is_focus = word_df['Perturbation_Type'] == 'Instruction_Word'
    elif focus == 'Claim':
        is_focus = word_df['Perturbation_Type'] == 'Query_Word'
    else:
        return

    instance_cols = ['Dataset', 'Model_Family', 'Model_Version', 'Topic', 'Intent', 'Instruction', 'Doc_ID', 'Doc_Type']
    
    focus_sums = word_df[is_focus].groupby(instance_cols)['Abs_Delta'].sum()
    total_sums = word_df.groupby(instance_cols)['Abs_Delta'].sum()

    rs_df = pd.DataFrame({f'{focus}_Sum': focus_sums, 'Total_Sum': total_sums}).fillna(0).reset_index()
    rs_df[metric_abbr] = rs_df[f'{focus}_Sum'] / (rs_df['Total_Sum'] + 1e-9)

    datasets = rs_df['Dataset'].unique()
    families = rs_df['Model_Family'].unique()

    version_palette = {
        "Base": "#FA7F6A", 
        "Homogeneous": "#4C72B0", 
        "Mixed": "#55A868", 
        "Mixed + Augmentation": "#C44E52"
    }

    hist_bins = np.linspace(0, 1, 21)

    for dataset in datasets:
        display_dataset = DATASET_DISPLAY.get(dataset, dataset) # Fallback to original string if not mapped
        
        for family in families:
            family_subset = rs_df[(rs_df['Dataset'] == dataset) & (rs_df['Model_Family'] == family)]
            
            if family_subset.empty:
                continue
            
            plot_dir = os.path.join(output_dir, dataset.replace(' ', '_').lower(), family.lower())
            os.makedirs(plot_dir, exist_ok=True)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
            
            subplots_info = [
                ("baseline", ["Base", "Homogeneous"], axes[0], "Baseline Impact"),
                ("mixture",  ["Homogeneous", "Mixed", "Mixed_Aug"], axes[1], "Mixture Ablation")
            ]

            for comp_name, comp_versions, ax, title_suffix in subplots_info:
                comp_subset = family_subset[family_subset['Model_Version'].isin(comp_versions)].copy()
                
                if comp_subset.empty or len(comp_subset['Model_Version'].unique()) < 1:
                    continue

                comp_subset['Model Version'] = comp_subset['Model_Version'].map(VERSION_DISPLAY)
                valid_hue_order = [VERSION_DISPLAY[v] for v in comp_versions if v in comp_subset['Model_Version'].unique()]

                sns.histplot(
                    data=comp_subset, x=metric_abbr, hue="Model Version",
                    hue_order=valid_hue_order, palette=version_palette, 
                    bins=hist_bins, element="step", stat="count", 
                    common_norm=False, alpha=0.3, ax=ax
                )

                ax.set_xlabel(f"Relative {focus} Sensitivity ({metric_abbr})")
                ax.set_title(f"{metric_abbr} | {display_dataset} | {family}\n({title_suffix})", fontsize=12)

                sns.move_legend(ax, "upper left")
                
                if ax == axes[0]:
                    ax.set_ylabel("Count")
                else:
                    ax.set_ylabel("")

            plt.tight_layout()
            save_path = os.path.join(plot_dir, f"{metric_abbr.lower()}_side_by_side.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

def generate_di_metrics(df, stance_keywords, output_dir):
    print("Generating Directional Impact (DI) Plots...")
    
    is_instr_perturbation = df['Perturbation_Type'] == 'Instruction_Word'
    is_stance_word = df['Removed_Text'].astype(str).str.lower().isin(stance_keywords)
    
    dim_df = df[is_instr_perturbation & is_stance_word].copy()

    instance_cols = ['Dataset', 'Model_Family', 'Model_Version', 'Intent', 'Doc_Type', 'Topic', 'Instruction']
    plot_df = dim_df.groupby(instance_cols)['First_Order_Delta'].mean().reset_index()
    plot_df = plot_df[plot_df['Doc_Type'].isin(['Relevant', 'Hard_Negative', 'Positive'])]

    datasets = plot_df['Dataset'].unique()
    families = plot_df['Model_Family'].unique()

    full_versions_order = ["Base", "Homogeneous", "Mixed", "Mixed_Aug"]

    for dataset in datasets:
        display_dataset = DATASET_DISPLAY.get(dataset, dataset)
        
        for family in families:
            family_subset = plot_df[(plot_df['Dataset'] == dataset) & (plot_df['Model_Family'] == family)].copy()
            
            if family_subset.empty:
                continue

            plot_dir = os.path.join(output_dir, dataset.replace(' ', '_').lower(), family.lower())
            os.makedirs(plot_dir, exist_ok=True)

            family_subset['Model Version'] = family_subset['Model_Version'].map(VERSION_DISPLAY)
            family_subset['Document Type'] = family_subset['Doc_Type'].map(DOC_DISPLAY)
            
            valid_x_order = [VERSION_DISPLAY[v] for v in full_versions_order if v in family_subset['Model_Version'].unique()]

            plt.figure(figsize=(10, 5))
            
            sns.violinplot(
                data=family_subset, x="Model Version", y="First_Order_Delta", 
                hue="Document Type", order=valid_x_order, split=True, inner="quartile",
                palette={"Relevant": "#2ca02c", "Positive": "#2ca02c", "Hard Negative": "#d62728"}
            )

            plt.axhline(0, color='black', linestyle='-', linewidth=1.2)
            plt.ylabel("Raw Delta (Similarity Change)")
            plt.xlabel("Model Version")
            
            plt.title(f"Directional Impact | {display_dataset} | {family}", fontsize=12)
            
            plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
            plt.tight_layout()
            
            save_path = os.path.join(plot_dir, f"di_all_versions.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate Word Ablation Figures (RIS, RCS, DI)")
    parser.add_argument("--results_dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"),
                        help="Root directory containing the evaluation subdirectories with word_ablation_results.csv files.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Path to save the generated figures. Defaults to results_dir/ablation_figures/")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "ablation_figures")
        
    os.makedirs(args.output_dir, exist_ok=True)

    # Set visualization theme
    sns.set_theme(style="whitegrid", rc={"axes.facecolor": PLOT_BG_COLOR, "figure.facecolor": PLOT_BG_COLOR})
    plt.rcParams.update({'font.size': FONT_SIZE, 'figure.dpi': 300, 'text.color': GLOBAL_TEXT_COLOR,
                         'axes.labelcolor': GLOBAL_TEXT_COLOR, 'xtick.color': GLOBAL_TEXT_COLOR, 'ytick.color': GLOBAL_TEXT_COLOR})

    # Dynamically load data from all subdirectories
    print(f"Scanning '{args.results_dir}' for word_ablation_results.csv files...")
    dataframes = []
    
    # We walk the directories to find CSVs. We assume the parent folder name maps to our datasets.
    for root, dirs, files in os.walk(args.results_dir):
        if "word_ablation_results.csv" in files:
            file_path = os.path.join(root, "word_ablation_results.csv")
            temp_df = pd.read_csv(file_path)
            dataframes.append(temp_df)
            print(f"  -> Loaded: {file_path}")

    if not dataframes:
        print("Error: No data files found. Please ensure ablation.py has generated results.")
        return

    df = pd.concat(dataframes, ignore_index=True)

    # Extract cleanly formatted Model Families and Versions
    print("Parsing model architectures and variants...")
    if 'Model' not in df.columns:
        print("Error: 'Model' column not found in data.")
        return

    # Family Extraction
    family_conditions = [
        df['Model'].str.contains('Qwen', na=False),
        df['Model'].str.contains('Instructor', na=False),
        df['Model'].str.contains('BGE', na=False)
    ]
    family_choices = ['Qwen3-Embedding-8B', 'Instructor-XL', 'BGE-Large']
    df['Model_Family'] = np.select(family_conditions, family_choices, default='Other')

    # Version Extraction (Order strictly matters)
    version_conditions = [
        df['Model'].str.contains('_base', case=False, na=False),
        df['Model'].str.contains('_homogeneous', case=False, na=False),
        df['Model'].str.contains('_mixed_aug', case=False, na=False),
        df['Model'].str.contains('_mixed', case=False, na=False)
    ]
    version_choices = ['Base', 'Homogeneous', 'Mixed_Aug', 'Mixed']
    df['Model_Version'] = np.select(version_conditions, version_choices, default='unknown')

    # Generate Plots
    generate_rs_metrics(df, focus="Instruction", output_dir=args.output_dir)
    generate_rs_metrics(df, focus="Claim", output_dir=args.output_dir)
    generate_di_metrics(df, STANCE_KEYWORDS, output_dir=args.output_dir)

    print(f"\nSuccess! All individual plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()