import os
import pandas as pd
import numpy as np
import argparse

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")

parser = argparse.ArgumentParser(description="Compute Word Ablation Metrics (RIS, RCS, DI)")
parser.add_argument("--results_dir", type=str, default=RESULTS_DIR,
                    help="Root directory containing the evaluation subdirectories with word_ablation_results.csv files.")
parser.add_argument("--output_csv", type=str, default=None,
                    help="Path to save the summary statistics CSV. Defaults to results_dir/ablation_summary_stats.csv")
args = parser.parse_args()

STANCE_KEYWORDS = {
    "benefits", "good", "agree", "positive", "downsides", "bad", "disagree", "negative",
    "backing", "favor", "validating", "corroborating", "upholding", "affirming", 
    "supports", "substantiating", "advocating", "verifying", "promoting", 
    "confirming", "justifying", "proving", "agreeing", "endorsing",
    "refuting", "against", "invalidating", "contradicting", "opposing", "challenging", 
    "undermines", "disputing", "arguing", "debunking", "dismissing", "disproving", 
    "attacking", "rejecting", "doubting", "criticisms"
}

# Metrics Calculations
def compute_rs_stats(df, focus):
    """Computes Relative Sensitivity (RS) for either Instruction (RIS) or Claim (RCS)."""
    metric_abbr = f"R{focus[0]}S"
    print(f"Computing {metric_abbr} ({focus} Sensitivity)...")
    
    # Isolate word perturbations
    word_df = df[df['Perturbation_Type'].isin(['Instruction_Word', 'Query_Word'])].copy()
    word_df['Abs_Delta'] = word_df['First_Order_Delta'].abs()

    if focus == 'Instruction':
        is_focus = word_df['Perturbation_Type'] == 'Instruction_Word'
    elif focus == 'Claim':
        is_focus = word_df['Perturbation_Type'] == 'Query_Word'
    else:
        is_focus = False

    instance_cols = ['Dataset', 'Model_Family', 'Model_Version', 'Topic', 'Intent', 'Instruction', 'Doc_ID', 'Doc_Type']
    
    # Calculate impact fraction per query
    focus_sums = word_df[is_focus].groupby(instance_cols)['Abs_Delta'].sum()
    total_sums = word_df.groupby(instance_cols)['Abs_Delta'].sum()

    rs_df = pd.DataFrame({f'{focus}_Sum': focus_sums, 'Total_Sum': total_sums}).fillna(0).reset_index()
    rs_df[metric_abbr] = rs_df[f'{focus}_Sum'] / (rs_df['Total_Sum'] + 1e-9)

    # Aggregate final statistics
    stats = rs_df.groupby(['Dataset', 'Model_Family', 'Model_Version'])[metric_abbr].agg(['mean', 'std']).reset_index()
    stats.rename(columns={'mean': 'Mean', 'std': 'Std'}, inplace=True)
    stats['Metric'] = metric_abbr
    stats['Doc_Type'] = 'All'
    
    return stats.to_dict('records')

def compute_di_stats(df, stance_keywords):
    """Computes Directional Impact (DI) based on explicit stance keywords."""
    print("Computing DI (Directional Impact)...")
    is_instr_perturbation = df['Perturbation_Type'] == 'Instruction_Word'
    is_stance_word = df['Removed_Text'].astype(str).str.lower().isin(stance_keywords)
    
    dim_df = df[is_instr_perturbation & is_stance_word].copy()

    instance_cols = ['Dataset', 'Model_Family', 'Model_Version', 'Intent', 'Doc_Type', 'Topic', 'Instruction']
    query_means = dim_df.groupby(instance_cols)['First_Order_Delta'].mean().reset_index()
    
    # Filter for standard document mappings
    query_means = query_means[query_means['Doc_Type'].isin(['Relevant', 'Hard_Negative', 'Positive'])]

    # Aggregate final statistics broken down by Document Type
    stats = query_means.groupby(['Dataset', 'Model_Family', 'Model_Version', 'Doc_Type'])['First_Order_Delta'].agg(['mean', 'std']).reset_index()
    stats.rename(columns={'mean': 'Mean', 'std': 'Std'}, inplace=True)
    stats['Metric'] = 'DI'
    
    return stats.to_dict('records')

def main():
    if args.output_csv is None:
        args.output_csv = os.path.join(args.results_dir, "ablation_summary_stats.csv")

    print(f"Scanning '{args.results_dir}' for word_ablation_results.csv files...")
    dataframes = []
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

    # Version Extraction (Order strictly matters to prevent 'Mixed' catching 'Mixed_Aug')
    version_conditions = [
        df['Model'].str.contains('_base', case=False, na=False),
        df['Model'].str.contains('_homogeneous', case=False, na=False),
        df['Model'].str.contains('_mixed_aug', case=False, na=False),
        df['Model'].str.contains('_mixed', case=False, na=False)
    ]
    version_choices = ['Base', 'Homogeneous', 'Mixed_Aug', 'Mixed']
    df['Model_Version'] = np.select(version_conditions, version_choices, default='unknown')

    # Generate Metrics
    stats_list = []
    stats_list.extend(compute_rs_stats(df, focus="Instruction"))
    stats_list.extend(compute_rs_stats(df, focus="Claim"))
    stats_list.extend(compute_di_stats(df, STANCE_KEYWORDS))

    # Save clean summary statistics
    stats_df = pd.DataFrame(stats_list)
    
    # Enforce standard column ordering
    cols = ['Metric', 'Dataset', 'Model_Family', 'Model_Version', 'Doc_Type', 'Mean', 'Std']
    stats_df = stats_df[[c for c in cols if c in stats_df.columns]]
    
    stats_df.to_csv(args.output_csv, index=False)
    print(f"\nSuccess! Summary statistics saved to: {args.output_csv}")

if __name__ == "__main__":
    main()