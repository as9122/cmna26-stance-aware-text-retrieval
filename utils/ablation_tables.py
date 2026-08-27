import os
import pandas as pd
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")

# Configuration via argparse
parser = argparse.ArgumentParser(description="Generate CEURART LaTeX tables from ablation metrics.")
parser.add_argument("--csv_path", type=str, 
                    default=os.path.join(RESULTS_DIR, "ablation_summary_stats.csv"), 
                    help="Path to the ablation_summary_stats.csv file.")
parser.add_argument("--output_dir", type=str, 
                    default=os.path.join(RESULTS_DIR, "latex_tables"), 
                    help="Where to save the generated .tex files.")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# The metrics script now outputs clean names, so we only need to map the strategies
STRATEGY_MAPPING = {
    'Base': 'Zero-Shot',
    'Homogeneous': 'Homogeneous',
    'Mixed': 'Mixed',
    'Mixed_Aug': 'Mixed + Aug'
}

FAMILY_ORDER = ['BGE-Large', 'Instructor-XL', 'Qwen3-Embedding-8B']
STRATEGY_ORDER = ['Base', 'Homogeneous', 'Mixed', 'Mixed_Aug']

def generate_ablation_tables():
    print("\n==========================================")
    print("Generating CEURART Ablation Tables...")
    print("==========================================")
    
    if not os.path.exists(args.csv_path):
        print(f"Error: Could not find data file at {args.csv_path}")
        return
        
    df = pd.read_csv(args.csv_path)
    
    def get_col_name(row):
        if row['Metric'] == 'DI':
            return 'DI (Pos)' if row['Doc_Type'] == 'Positive' else 'DI (Hard Neg)'
        return row['Metric'] # RIS or RCS
        
    df['Table_Col'] = df.apply(get_col_name, axis=1)
    
    col_order = ['RIS', 'RCS', 'DI (Pos)', 'DI (Hard Neg)']
    datasets = df['Dataset'].unique()
    all_tables_lines = []
    
    for dataset in datasets:
        d_df = df[df['Dataset'] == dataset]
        
        # Escape percentage symbol for LaTeX
        safe_dataset_name = dataset.replace("%", "\\%")
        
        caption = f"Ablation metrics on the \\textbf{{{safe_dataset_name}}} dataset. Results present the mean ($\\mu$) and standard deviation ($\\sigma$). Metrics include Relative Information Score (RIS), Relative Contrastive Score (RCS), and Document Impact (DI) calculated for positive and hard negative pairs."
        
        lines = [
            "\\begin{table*}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{tab:ablation_{dataset.replace(' ', '_').replace('%', '').lower()}}}",
            "\\renewcommand{\\arraystretch}{1.2}",
            "\\begin{tabular}{ll cccc}",
            "\\toprule[1.5pt]",
            "\\textbf{Architecture} & \\textbf{Strategy} & \\textbf{RIS} & \\textbf{RCS} & \\textbf{DI (Pos)} & \\textbf{DI (Hard Neg)} \\\\",
            "\\midrule[1.5pt]"
        ]
        
        first_family = True
        for family in FAMILY_ORDER:
            fam_df = d_df[d_df['Model_Family'] == family]
            if fam_df.empty:
                continue
                
            if not first_family:
                lines.append("\\addlinespace[1.5em]")
            first_family = False
            
            for s_idx, strat_key in enumerate(STRATEGY_ORDER):
                strat_df = fam_df[fam_df['Model_Version'] == strat_key]
                strat_str = STRATEGY_MAPPING.get(strat_key, strat_key)
                
                # Multirow architecture name only on the first row of the block
                col1 = f"\\multirow{{{len(STRATEGY_ORDER)}}}{{*}}{{{family}}}" if s_idx == 0 else ""
                
                # Extract values for the 4 metric columns
                row_vals = []
                for col_name in col_order:
                    val_df = strat_df[strat_df['Table_Col'] == col_name]
                    if not val_df.empty:
                        mean_val = val_df['Mean'].iloc[0]
                        std_val = val_df['Std'].iloc[0]
                        row_vals.append(f"${mean_val:.3f}\\pm{std_val:.2f}$")
                    else:
                        row_vals.append("$-$")
                        
                metrics_joined = " & ".join(row_vals)
                lines.append(f"{col1} & {strat_str} & {metrics_joined} \\\\")
                
        lines.append("\\bottomrule[1.5pt]")
        lines.append("\\end{tabular}")
        lines.append("\\end{table*}")
        
        # Save individual dataset table
        clean_filename = f"table_ablation_{dataset.replace(' ', '_').replace('%', '').lower()}"
        out_path = os.path.join(args.output_dir, f"{clean_filename}.tex")
        with open(out_path, 'w') as f:
            f.write("\n".join(lines))
        print(f"  -> Saved Table: {out_path}")
        
        all_tables_lines.extend(lines)
        all_tables_lines.append("\n\\clearpage\n\\vspace{2em}\n")

    # Save combined file
    if all_tables_lines:
        combined_path = os.path.join(args.output_dir, "all_ablation_tables_combined.tex")
        with open(combined_path, 'w') as f:
            f.write("\n".join(all_tables_lines))
        print(f"\n  -> Saved Combined File: {combined_path}")

if __name__ == "__main__":
    generate_ablation_tables()