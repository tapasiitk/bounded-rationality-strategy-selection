#!/usr/bin/env python3
"""
Complete code to generate model comparison figure from stimulus and response data.

This script:
1. Loads stimulus and response data
2. Implements both computational models
3. Calculates observed choice proportions with confidence intervals
4. Generates model predictions
5. Creates publication-quality figure
6. Calculates goodness-of-fit metrics

Author: Tapas Rath
Date: September 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata, beta
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300
})

def load_and_prepare_data(stimuli_file, response_file):
    """Load and merge stimulus and response data."""
    print("Loading data...")
    
    # Load data files
    stimuli = pd.read_csv(stimuli_file)
    responses = pd.read_csv(response_file)
    
    # Exclude catch trials
    stimuli = stimuli[stimuli['trial_type'] != 'catch']
    
    # Prepare stimulus data - create lists of A, B, C values per trial
    stim_summ = (
        stimuli.groupby('trial')
        .agg({
            'A': lambda x: list(x),
            'B': lambda x: list(x), 
            'C': lambda x: list(x),
            'target': 'first'
        })
        .reset_index()
        .rename(columns={'A':'A_frames', 'B':'B_frames', 'C':'C_frames'})
    )
    
    # Prepare response data - one response per trial per participant
    resp_summ = (
        responses.groupby(['userId', 'trial']).agg({
            'response': 'first',
            'target': 'first'
        }).reset_index()
    )
    
    # Merge data
    df = resp_summ.merge(stim_summ, on='trial', how='left', suffixes=('', '_stim'))
    df = df.dropna(subset=['A_frames','B_frames','C_frames','response'])
    
    print(f"Data loaded: {len(df)} trials from {df['userId'].nunique()} participants")
    
    return df

def rank_based_model_prediction(A, B, C, w1, w2, w3, noise):
    """Rank-based salience model prediction for single trial."""
    P = np.zeros(3)
    
    for i in range(len(A)):
        vals = [A[i], B[i], C[i]]
        # Rank 0=best, 2=worst  
        ranks = rankdata([-v for v in vals], method='ordinal') - 1
        weights = [w1, w2, w3]
        
        for j in range(3):
            P[j] += vals[j] * weights[ranks[j]]
    
    # Apply softmax
    P_normalized = P - np.max(P)
    exp_P = np.exp(P_normalized / noise)
    return exp_P / exp_P.sum()

def optimise_strategy(A, B, C, beta):
    """Optimization strategy - sum all values."""
    totals = np.array([sum(A), sum(B), sum(C)])
    exp_S = np.exp(beta * (totals - totals.max()))
    return exp_S / exp_S.sum()

def satisficing_strategy(A, B, C, alpha):
    """Satisficing strategy - pairwise comparisons."""
    wins = np.zeros(3)
    
    for i in range(len(A)):
        vals = [A[i], B[i], C[i]]
        for j in range(3):
            wins[j] += sum(vals[j] > vals[k] for k in range(3) if k != j)
    
    exp_W = np.exp(alpha * (wins - wins.max()))
    return exp_W / exp_W.sum()

def meta_model(A, B, C, lmbda, beta, alpha):
    """Effort-based meta-cognitive model - probabilistic mixture."""
    p_opt = optimise_strategy(A, B, C, beta)
    p_sat = satisficing_strategy(A, B, C, alpha)
    return lmbda * p_opt + (1 - lmbda) * p_sat

def calculate_observed_proportions_with_ci(data):
    """Calculate observed choice proportions with 95% confidence intervals."""
    choice_counts = data['response'].value_counts()
    n_total = len(data)
    
    results = {}
    for choice in ['A', 'B', 'C']:
        count = choice_counts.get(choice, 0)
        prop = count / n_total
        
        # Wilson score interval for binomial proportion
        if n_total > 0:
            # Use beta distribution for CI (more accurate for small counts)
            alpha_param = count + 1
            beta_param = n_total - count + 1
            ci_low = beta.ppf(0.025, alpha_param, beta_param)
            ci_high = beta.ppf(0.975, alpha_param, beta_param)
        else:
            ci_low = ci_high = prop
            
        results[choice] = {
            'prop': prop,
            'ci_low': max(0, ci_low),
            'ci_high': min(1, ci_high),
            'ci_error': [prop - max(0, ci_low), min(1, ci_high) - prop]
        }
    
    return results

def get_model_predictions(data, model_func, params):
    """Get average model predictions for a condition."""
    predictions = []
    
    for _, row in data.iterrows():
        if model_func == rank_based_model_prediction:
            pred = model_func(row['A_frames'], row['B_frames'], row['C_frames'], *params)
        else:  # meta_model
            pred = model_func(row['A_frames'], row['B_frames'], row['C_frames'], *params)
        predictions.append(pred)
    
    # Average predictions across all trials
    avg_pred = np.mean(predictions, axis=0)
    return {'A': avg_pred[0], 'B': avg_pred[1], 'C': avg_pred[2]}

def calculate_goodness_of_fit(observed, predicted):
    """Calculate correlation and RMSE between observed and predicted values."""
    from scipy.stats import pearsonr
    
    obs_vals = list(observed.values())
    pred_vals = list(predicted.values())
    
    r, p_val = pearsonr(obs_vals, pred_vals)
    rmse = np.sqrt(np.mean((np.array(obs_vals) - np.array(pred_vals))**2))
    
    return r, rmse

def create_model_comparison_figure(df, rank_params, meta_params, save_path='figures/model_comparison_figure.png'):
    """Create the complete model comparison figure."""
    
    print("Calculating observed proportions and model predictions...")
    
    # Separate data by target condition
    df_A_target = df[df['target'] == 'A'].copy()
    df_C_target = df[df['target'] == 'C'].copy()
    
    # Calculate observed proportions with confidence intervals
    obs_A_target_ci = calculate_observed_proportions_with_ci(df_A_target)
    obs_C_target_ci = calculate_observed_proportions_with_ci(df_C_target)
    
    # Get model predictions
    rank_pred_A = get_model_predictions(df_A_target, rank_based_model_prediction, rank_params)
    meta_pred_A = get_model_predictions(df_A_target, meta_model, meta_params)
    
    rank_pred_C = get_model_predictions(df_C_target, rank_based_model_prediction, rank_params)
    meta_pred_C = get_model_predictions(df_C_target, meta_model, meta_params)
    
    # Calculate goodness-of-fit metrics
    obs_all = [obs_A_target_ci[c]['prop'] for c in ['A','B','C']] + [obs_C_target_ci[c]['prop'] for c in ['A','B','C']]
    meta_all = [meta_pred_A[c] for c in ['A','B','C']] + [meta_pred_C[c] for c in ['A','B','C']]
    rank_all = [rank_pred_A[c] for c in ['A','B','C']] + [rank_pred_C[c] for c in ['A','B','C']]
    
    r_meta, rmse_meta = calculate_goodness_of_fit({i: obs_all[i] for i in range(6)}, 
                                                  {i: meta_all[i] for i in range(6)})
    r_rank, rmse_rank = calculate_goodness_of_fit({i: obs_all[i] for i in range(6)}, 
                                                  {i: rank_all[i] for i in range(6)})
    
    print(f"Goodness of fit - Effort-based: r={r_meta:.3f}, RMSE={rmse_meta:.3f}")
    print(f"Goodness of fit - Rank-based: r={r_rank:.3f}, RMSE={rmse_rank:.3f}")
    
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('Model Comparison: Observed vs Predicted Choice Distributions', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Define positions and colors
    x_pos = np.arange(3)
    width = 0.25
    choices = ['A', 'B', 'C']
    colors = ['#2C3E50', '#3498DB', '#E74C3C']  # Dark blue, light blue, red
    
    # Panel 1: When A is Target
    ax1.set_title('When A is Target', fontsize=14, fontweight='bold')
    
    obs_data_A = [obs_A_target_ci[c]['prop'] for c in choices]
    effort_data_A = [meta_pred_A[c] for c in choices]
    rank_data_A = [rank_pred_A[c] for c in choices]
    
    # Error bars for observed data
    obs_errors_A = [[obs_A_target_ci[c]['ci_error'][0] for c in choices], 
                    [obs_A_target_ci[c]['ci_error'][1] for c in choices]]
    
    # Create bars
    bars1_1 = ax1.bar(x_pos - width, obs_data_A, width, label='Observed', 
                      color=colors[0], alpha=0.8, edgecolor='black', linewidth=1.2,
                      yerr=obs_errors_A, capsize=5, error_kw={'elinewidth': 2})
    bars1_2 = ax1.bar(x_pos, effort_data_A, width, label='Effort-Based Model', 
                      color=colors[1], alpha=0.7, edgecolor='black', linewidth=1.2)
    bars1_3 = ax1.bar(x_pos + width, rank_data_A, width, label='Rank-Based Model', 
                      color=colors[2], alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # Panel 2: When C is Target
    ax2.set_title('When C is Target', fontsize=14, fontweight='bold')
    
    obs_data_C = [obs_C_target_ci[c]['prop'] for c in choices]
    effort_data_C = [meta_pred_C[c] for c in choices]
    rank_data_C = [rank_pred_C[c] for c in choices]
    
    obs_errors_C = [[obs_C_target_ci[c]['ci_error'][0] for c in choices], 
                    [obs_C_target_ci[c]['ci_error'][1] for c in choices]]
    
    bars2_1 = ax2.bar(x_pos - width, obs_data_C, width, 
                      color=colors[0], alpha=0.8, edgecolor='black', linewidth=1.2,
                      yerr=obs_errors_C, capsize=5, error_kw={'elinewidth': 2})
    bars2_2 = ax2.bar(x_pos, effort_data_C, width, 
                      color=colors[1], alpha=0.7, edgecolor='black', linewidth=1.2)
    bars2_3 = ax2.bar(x_pos + width, rank_data_C, width, 
                      color=colors[2], alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # Format both panels
    for ax in [ax1, ax2]:
        ax.set_ylabel('Choice Probability', fontsize=12)
        ax.set_xlabel('Choice Option', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(choices, fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Add legend
    fig.legend(['Observed (95% CI)', 'Effort-Based Model', 'Rank-Based Model'], 
               loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved as '{save_path}'")
    
    # Print results summary
    print(f"\n=== RESULTS SUMMARY ===")
    print(f"When A is target - Observed: A={obs_data_A[0]:.3f}, B={obs_data_A[1]:.3f}, C={obs_data_A[2]:.3f}")
    print(f"When A is target - Effort-based: A={effort_data_A[0]:.3f}, B={effort_data_A[1]:.3f}, C={effort_data_A[2]:.3f}")
    print(f"When A is target - Rank-based: A={rank_data_A[0]:.3f}, B={rank_data_A[1]:.3f}, C={rank_data_A[2]:.3f}")
    
    print(f"\nWhen C is target - Observed: A={obs_data_C[0]:.3f}, B={obs_data_C[1]:.3f}, C={obs_data_C[2]:.3f}")
    print(f"When C is target - Effort-based: A={effort_data_C[0]:.3f}, B={effort_data_C[1]:.3f}, C={effort_data_C[2]:.3f}")
    print(f"When C is target - Rank-based: A={rank_data_C[0]:.3f}, B={rank_data_C[1]:.3f}, C={rank_data_C[2]:.3f}")
    
    # Save data for LaTeX/other use
    plot_data = {
        'condition': ['A_target'] * 3 + ['C_target'] * 3,
        'choice': ['A', 'B', 'C'] * 2,
        'observed': obs_data_A + obs_data_C,
        'effort_based': effort_data_A + effort_data_C,
        'rank_based': rank_data_A + rank_data_C
    }
    
    plot_df = pd.DataFrame(plot_data)
    plot_df.to_csv('results/model_comparison_data.csv', index=False)
    print(f"Data saved to 'results/model_comparison_data.csv'")
    
    plt.show()
    
    return fig, (r_meta, rmse_meta, r_rank, rmse_rank)

def main():
    """Main function to generate the complete figure."""
    
    # File paths (adjust these to your file locations)
    STIMULI_FILE = 'data/stimuli.csv'
    RESPONSE_FILE = 'data/clean_data.csv'
    
    # Your fitted model parameters
    RANK_PARAMS = [1.00, 0.50, 0.10, 10.0]  # w1, w2, w3, noise
    META_PARAMS = [0.974, 0.009, 5.000]     # lambda, beta, alpha
    
    try:
        # Load and prepare data
        df = load_and_prepare_data(STIMULI_FILE, RESPONSE_FILE)
        
        # Create the figure
        fig, metrics = create_model_comparison_figure(df, RANK_PARAMS, META_PARAMS)
        
        r_meta, rmse_meta, r_rank, rmse_rank = metrics
        
        print(f"\n=== FIGURE CAPTION ===")
        print(f"Figure 1. Model comparison showing observed choice distributions and model predictions.")
        print(f"Left panel shows choices when A was the target; right panel shows choices when C was the target.")
        print(f"Dark bars represent observed data with 95% confidence intervals (n = {len(df)} trials),")
        print(f"light blue bars show effort-based model predictions, and red bars show rank-based model predictions.")
        print(f"The effort-based model closely matches observed data (r = {r_meta:.3f}, RMSE = {rmse_meta:.3f}),")
        print(f"while the rank-based model shows substantial deviations (r = {r_rank:.3f}, RMSE = {rmse_rank:.3f}),")
        print(f"consistent with the large difference in model fit (ΔAIC = 23,175).")
        print(f"The extreme predictions of the rank-based model demonstrate why this model fails for commensurable attributes.")
        
        print(f"\n=== SUCCESS ===")
        print(f"Figure generation complete! Files created:")
        print(f"- model_comparison_figure.png (high-resolution figure)")
        print(f"- model_comparison_data.csv (data for LaTeX/other use)")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. Please check file paths:")
        print(f"- {STIMULI_FILE}")
        print(f"- {RESPONSE_FILE}")
        print(f"Make sure these files are in the same directory as this script.")
        
    except Exception as e:
        print(f"Error: {e}")
        print(f"Please check your data files and try again.")

if __name__ == "__main__":
    main()