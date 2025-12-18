"""
Complete Python Implementation for Model Fitting and Comparison
Rank-Based vs Pairwise Comparison Models for Multi-Attribute RSVP Task

Author: Analysis for testing rank-based models in multi-attribute numerical RSVP task
Date: September 2025

This script implements:
1. Data preparation and merging
2. Rank-based model (Tsetsos et al., 2012) - attribute-wise processing
3. Pairwise comparison model - item-wise processing with commensurate attributes
4. Model fitting using grid search
5. Model comparison using AIC
6. Visualization and results reporting

Usage:
    python model_fitting_complete_code.py

Requirements:
    - pandas
    - numpy
    - scipy
    - matplotlib
    - seaborn
"""

import pandas as pd
import numpy as np
from scipy.stats import rankdata
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: DATA PREPARATION
# ============================================================================

def load_and_prepare_data(stimuli_file, response_file):
    """
    Load and merge stimulus and response data for model fitting
    
    Parameters:
    - stimuli_file: Path to CSV with stimulus information
    - response_file: Path to CSV with participant responses
    
    Returns:
    - merged_data: DataFrame ready for model fitting
    """
    print("=== STEP 1: DATA PREPARATION ===")
    
    # Load data
    stimuli_df = pd.read_csv(stimuli_file)
    response_df = pd.read_csv(response_file)
    
    print(f"Loaded {len(stimuli_df)} stimulus rows and {len(response_df)} response rows")
    
    # Create stimulus summary with frame-by-frame information
    stimulus_summary = stimuli_df.groupby('trial').agg({
        'trial_type': 'first',
        'target': 'first', 
        'A': lambda x: x.tolist(),  # All 24 frame values for A
        'B': lambda x: x.tolist(),  # All 24 frame values for B
        'C': lambda x: x.tolist(),  # All 24 frame values for C
        'color': lambda x: x.tolist(), # Color sequence (Red/Blue)
        'arrange': 'first'
    }).reset_index()
    
    # Rename columns to avoid conflicts
    stimulus_clean = stimulus_summary.rename(columns={
        'A': 'A_frames',
        'B': 'B_frames',
        'C': 'C_frames', 
        'color': 'color_frames'
    })
    
    # Merge with response data
    final_data = response_df[['userId', 'trial', 'response', 'target']].merge(
        stimulus_clean[['trial', 'A_frames', 'B_frames', 'C_frames', 'color_frames', 'trial_type', 'arrange']],
        on='trial'
    )
    
    print(f"Successfully merged data: {len(final_data)} trials from {final_data['userId'].nunique()} participants")
    print(f"Response distribution: {final_data['response'].value_counts().to_dict()}")
    
    # Data integrity checks
    assert all(len(row) == 24 for row in final_data['A_frames']), "All A sequences should have 24 frames"
    assert all(len(row) == 24 for row in final_data['B_frames']), "All B sequences should have 24 frames"  
    assert all(len(row) == 24 for row in final_data['C_frames']), "All C sequences should have 24 frames"
    
    print("Data integrity checks passed ✓")
    
    return final_data

# ============================================================================
# STEP 2: MODEL IMPLEMENTATIONS
# ============================================================================

def rank_based_model_prediction(A_frames, B_frames, C_frames, w_rank1, w_rank2, w_rank3, noise_std=100.0):
    """
    Implement Tsetsos et al. (2012) rank-based model
    
    This model processes information attribute-wise within each frame:
    - For each frame, ranks the three values (A, B, C) 
    - Applies rank-dependent weights (w_rank1 > w_rank2 > w_rank3)
    - Accumulates weighted values across all 24 frames
    - Returns choice probabilities using softmax
    
    Parameters:
    - A_frames, B_frames, C_frames: Lists of 24 values each
    - w_rank1, w_rank2, w_rank3: Rank weights (1st=highest, 2nd=middle, 3rd=lowest)
    - noise_std: Standard deviation for softmax temperature
    
    Returns:
    - probabilities: Array of [P(A), P(B), P(C)]
    """
    
    # Initialize preference accumulators
    P_A = P_B = P_C = 0.0
    
    # Process each frame independently
    for frame in range(24):
        # Get values for this frame
        values = [A_frames[frame], B_frames[frame], C_frames[frame]]
        
        # Rank values (1=highest, 2=middle, 3=lowest)
        # Use negative values to get descending rank order
        ranks = rankdata([-v for v in values], method='ordinal')
        
        # Define rank weights
        weights = [w_rank1, w_rank2, w_rank3]
        
        # Apply rank-dependent weighting and accumulate
        P_A += values[0] * weights[ranks[0] - 1]  # -1 because ranks are 1-indexed
        P_B += values[1] * weights[ranks[1] - 1]
        P_C += values[2] * weights[ranks[2] - 1]
    
    # Convert accumulators to choice probabilities using softmax
    accumulator_values = np.array([P_A, P_B, P_C])
    
    # Numerical stability: subtract max before exponentiating
    max_val = np.max(accumulator_values)
    stable_values = (accumulator_values - max_val) / noise_std
    exp_values = np.exp(stable_values)
    probabilities = exp_values / np.sum(exp_values)
    
    return probabilities

def pairwise_comparison_model_prediction(A_frames, B_frames, C_frames, color_frames, noise_std=100.0):
    """
    Implement pairwise comparison model with commensurate attributes
    
    This model processes information item-wise:
    - First separates frames by attribute (Red vs Blue)
    - Sums within each attribute across all frames
    - Combines attributes (Red + Blue totals) since they're commensurate
    - Makes probabilistic choices based on combined utilities
    
    Parameters:
    - A_frames, B_frames, C_frames: Lists of 24 values each
    - color_frames: List of 24 color labels ('Red' or 'Blue')
    - noise_std: Standard deviation for softmax temperature
    
    Returns:
    - probabilities: Array of [P(A), P(B), P(C)]
    """
    
    # Separate and sum by attribute (Red vs Blue)
    A_red_total = sum(A_frames[i] for i in range(24) if color_frames[i] == 'Red')
    A_blue_total = sum(A_frames[i] for i in range(24) if color_frames[i] == 'Blue')
    A_total = A_red_total + A_blue_total  # Combine commensurate attributes
    
    B_red_total = sum(B_frames[i] for i in range(24) if color_frames[i] == 'Red')
    B_blue_total = sum(B_frames[i] for i in range(24) if color_frames[i] == 'Blue')
    B_total = B_red_total + B_blue_total
    
    C_red_total = sum(C_frames[i] for i in range(24) if color_frames[i] == 'Red')
    C_blue_total = sum(C_frames[i] for i in range(24) if color_frames[i] == 'Blue')
    C_total = C_red_total + C_blue_total
    
    # Convert to choice probabilities using softmax
    utilities = np.array([A_total, B_total, C_total])
    
    # Numerical stability
    max_util = np.max(utilities)
    stable_utils = (utilities - max_util) / noise_std
    exp_utils = np.exp(stable_utils)
    probabilities = exp_utils / np.sum(exp_utils)
    
    return probabilities

# ============================================================================
# STEP 3: MODEL FITTING FUNCTIONS
# ============================================================================

def fit_rank_based_model(data, verbose=True):
    """
    Fit rank-based model to data using grid search
    
    Parameters:
    - data: DataFrame with trial data
    - verbose: Whether to print progress
    
    Returns:
    - Dictionary with best parameters, negative log-likelihood, and metadata
    """
    
    if verbose:
        print("=== FITTING RANK-BASED MODEL ===")
        print("Searching parameter space...")
    
    choice_to_index = {'A': 0, 'B': 1, 'C': 2}
    
    best_nll = float('inf')
    best_params = None
    
    # Define parameter search space
    w1_values = [0.8, 1.0, 1.2, 1.5, 2.0]
    w2_values = [0.3, 0.5, 0.7, 0.9]
    w3_values = [0.1, 0.2, 0.3]
    noise_values = [20, 50, 100, 150]
    
    total_combinations = len(w1_values) * len(w2_values) * len(w3_values) * len(noise_values)
    tested = 0
    
    # Grid search over parameter space
    for w1 in w1_values:
        for w2 in w2_values:
            for w3 in w3_values:
                for noise in noise_values:
                    tested += 1
                    
                    # Progress update
                    if verbose and tested % 50 == 0:
                        print(f"  Tested {tested}/{total_combinations} combinations...")
                    
                    # Constraint: w1 > w2 > w3 (rank ordering)
                    if w1 > w2 > w3:
                        try:
                            nll = 0.0
                            
                            # Calculate negative log-likelihood
                            for _, row in data.iterrows():
                                probs = rank_based_model_prediction(
                                    row['A_frames'], row['B_frames'], row['C_frames'],
                                    w1, w2, w3, noise
                                )
                                
                                # Get actual choice index
                                actual_idx = choice_to_index[row['response']]
                                
                                # Add to negative log-likelihood
                                prob = max(probs[actual_idx], 1e-10)  # Prevent log(0)
                                nll -= np.log(prob)
                            
                            # Update best parameters if this is better
                            if nll < best_nll:
                                best_nll = nll
                                best_params = [w1, w2, w3, noise]
                                
                        except Exception as e:
                            if verbose:
                                print(f"    Error with params [{w1}, {w2}, {w3}, {noise}]: {e}")
                            continue
    
    if verbose:
        if best_params:
            print(f"Best parameters found: w1={best_params[0]:.3f}, w2={best_params[1]:.3f}, w3={best_params[2]:.3f}, noise={best_params[3]:.1f}")
            print(f"Negative log-likelihood: {best_nll:.2f}")
        else:
            print("No valid parameters found!")
    
    return {
        'model_name': 'rank_based',
        'params': best_params,
        'param_names': ['w_rank1', 'w_rank2', 'w_rank3', 'noise_std'],
        'nll': best_nll,
        'n_params': 4,
        'n_trials': len(data)
    }

def fit_pairwise_model(data, verbose=True):
    """
    Fit pairwise comparison model to data using grid search
    
    Parameters:
    - data: DataFrame with trial data  
    - verbose: Whether to print progress
    
    Returns:
    - Dictionary with best parameters, negative log-likelihood, and metadata
    """
    
    if verbose:
        print("=== FITTING PAIRWISE COMPARISON MODEL ===")
        print("Searching parameter space...")
    
    choice_to_index = {'A': 0, 'B': 1, 'C': 2}
    
    best_nll = float('inf')
    best_params = None
    
    # Simpler parameter space (only noise parameter)
    noise_values = [10, 20, 50, 100, 150, 200]
    
    for noise in noise_values:
        if verbose:
            print(f"  Testing noise_std = {noise}")
            
        try:
            nll = 0.0
            
            # Calculate negative log-likelihood
            for _, row in data.iterrows():
                probs = pairwise_comparison_model_prediction(
                    row['A_frames'], row['B_frames'], row['C_frames'],
                    row['color_frames'], noise
                )
                
                # Get actual choice index
                actual_idx = choice_to_index[row['response']]
                
                # Add to negative log-likelihood
                prob = max(probs[actual_idx], 1e-10)  # Prevent log(0)
                nll -= np.log(prob)
            
            # Update best parameters if this is better
            if nll < best_nll:
                best_nll = nll
                best_params = [noise]
                
        except Exception as e:
            if verbose:
                print(f"    Error with noise={noise}: {e}")
            continue
    
    if verbose:
        if best_params:
            print(f"Best parameters found: noise_std={best_params[0]:.1f}")
            print(f"Negative log-likelihood: {best_nll:.2f}")
        else:
            print("No valid parameters found!")
    
    return {
        'model_name': 'pairwise',
        'params': best_params,
        'param_names': ['noise_std'],
        'nll': best_nll,
        'n_params': 1,
        'n_trials': len(data)
    }

# ============================================================================
# STEP 4: MODEL COMPARISON
# ============================================================================

def compare_models(rank_result, pairwise_result, verbose=True):
    """
    Compare models using AIC and other criteria
    
    Parameters:
    - rank_result: Results from rank-based model fitting
    - pairwise_result: Results from pairwise model fitting
    - verbose: Whether to print detailed comparison
    
    Returns:
    - Dictionary with comparison results
    """
    
    if verbose:
        print("=== MODEL COMPARISON ===")
    
    # Calculate AIC for both models
    # AIC = 2k + 2*NLL (where k = number of parameters)
    rank_aic = 2 * rank_result['n_params'] + 2 * rank_result['nll']
    pairwise_aic = 2 * pairwise_result['n_params'] + 2 * pairwise_result['nll']
    
    delta_aic = abs(rank_aic - pairwise_aic)
    better_model = 'rank_based' if rank_aic < pairwise_aic else 'pairwise'
    
    # Interpret evidence strength (Burnham & Anderson, 2002)
    if delta_aic > 10:
        evidence = "Very strong"
    elif delta_aic > 4:
        evidence = "Strong"
    elif delta_aic > 2:
        evidence = "Moderate"
    else:
        evidence = "Weak"
    
    if verbose:
        print(f"Rank-based model:")
        print(f"  AIC: {rank_aic:.2f}")
        print(f"  Parameters: {rank_result['n_params']}")
        print(f"  Negative log-likelihood: {rank_result['nll']:.2f}")
        
        print(f"\nPairwise model:")
        print(f"  AIC: {pairwise_aic:.2f}")
        print(f"  Parameters: {pairwise_result['n_params']}")
        print(f"  Negative log-likelihood: {pairwise_result['nll']:.2f}")
        
        print(f"\nComparison:")
        print(f"  Delta AIC: {delta_aic:.2f}")
        print(f"  Better model: {better_model}")
        print(f"  Evidence strength: {evidence}")
        
        if better_model == 'pairwise':
            print(f"  → Pairwise model wins by {rank_aic - pairwise_aic:.1f} AIC points")
        else:
            print(f"  → Rank-based model wins by {pairwise_aic - rank_aic:.1f} AIC points")
    
    return {
        'rank_aic': rank_aic,
        'pairwise_aic': pairwise_aic,
        'delta_aic': delta_aic,
        'better_model': better_model,
        'evidence_strength': evidence,
        'rank_result': rank_result,
        'pairwise_result': pairwise_result
    }

# ============================================================================
# STEP 5: PREDICTION ACCURACY AND VALIDATION
# ============================================================================

def calculate_prediction_accuracy(data, rank_result, pairwise_result, n_test=100):
    """
    Calculate prediction accuracy for both models
    
    Parameters:
    - data: DataFrame with trial data
    - rank_result: Fitted rank-based model results
    - pairwise_result: Fitted pairwise model results  
    - n_test: Number of trials to test on
    
    Returns:
    - Dictionary with accuracy results
    """
    
    print(f"=== PREDICTION ACCURACY (Testing on {n_test} trials) ===")
    
    # Use a sample of the data for testing
    test_data = data.sample(n=min(n_test, len(data)), random_state=42)
    
    correct_rank = 0
    correct_pairwise = 0
    
    choice_to_index = {'A': 0, 'B': 1, 'C': 2}
    index_to_choice = {0: 'A', 1: 'B', 2: 'C'}
    
    for _, row in test_data.iterrows():
        actual_choice = row['response']
        
        # Rank-based model prediction
        if rank_result['params']:
            w1, w2, w3, noise = rank_result['params']
            rank_probs = rank_based_model_prediction(
                row['A_frames'], row['B_frames'], row['C_frames'],
                w1, w2, w3, noise
            )
            rank_pred = index_to_choice[np.argmax(rank_probs)]
            if rank_pred == actual_choice:
                correct_rank += 1
        
        # Pairwise model prediction
        if pairwise_result['params']:
            noise = pairwise_result['params'][0]
            pair_probs = pairwise_comparison_model_prediction(
                row['A_frames'], row['B_frames'], row['C_frames'],
                row['color_frames'], noise
            )
            pair_pred = index_to_choice[np.argmax(pair_probs)]
            if pair_pred == actual_choice:
                correct_pairwise += 1
    
    rank_accuracy = (correct_rank / n_test) * 100 if rank_result['params'] else 0
    pairwise_accuracy = (correct_pairwise / n_test) * 100 if pairwise_result['params'] else 0
    
    print(f"Rank-based accuracy: {correct_rank}/{n_test} ({rank_accuracy:.1f}%)")
    print(f"Pairwise accuracy: {correct_pairwise}/{n_test} ({pairwise_accuracy:.1f}%)")
    
    return {
        'rank_accuracy': rank_accuracy,
        'pairwise_accuracy': pairwise_accuracy,
        'n_test': n_test
    }

# ============================================================================
# STEP 6: VISUALIZATION FUNCTIONS
# ============================================================================

def create_model_comparison_plot(comparison_results):
    """
    Create visualization comparing model fits
    """
    
    print("=== CREATING VISUALIZATION ===")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: AIC Comparison
    models = ['Rank-based', 'Pairwise']
    aics = [comparison_results['rank_aic'], comparison_results['pairwise_aic']]
    colors = ['lightcoral' if comparison_results['better_model'] == 'pairwise' else 'lightblue',
              'lightblue' if comparison_results['better_model'] == 'pairwise' else 'lightcoral']
    
    bars = axes[0].bar(models, aics, color=colors)
    axes[0].set_ylabel('AIC (lower = better)')
    axes[0].set_title('Model Comparison: AIC')
    
    # Add AIC values on bars
    for bar, aic in zip(bars, aics):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{aic:.1f}', ha='center', va='bottom')
    
    # Plot 2: Parameter Values
    if comparison_results['rank_result']['params'] and comparison_results['pairwise_result']['params']:
        rank_params = comparison_results['rank_result']['params']
        
        # Rank-based parameters
        param_names = ['w_rank1', 'w_rank2', 'w_rank3', 'noise_std']
        param_values = rank_params
        
        axes[1].bar(param_names, param_values, color='lightcoral', alpha=0.7)
        axes[1].set_ylabel('Parameter Value')
        axes[1].set_title('Rank-based Model Parameters')
        axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Plot saved as 'model_comparison_results.png'")

# ============================================================================
# STEP 7: MAIN ANALYSIS FUNCTION
# ============================================================================

def run_complete_analysis(stimuli_file, response_file, save_results=True):
    """
    Run complete model fitting and comparison analysis
    
    Parameters:
    - stimuli_file: Path to stimulus CSV file
    - response_file: Path to response CSV file
    - save_results: Whether to save results to files
    
    Returns:
    - Dictionary with all analysis results
    """
    
    print("="*60)
    print("COMPLETE MODEL FITTING AND COMPARISON ANALYSIS")
    print("="*60)
    
    # Step 1: Load and prepare data
    try:
        data = load_and_prepare_data(stimuli_file, response_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Step 2: Fit both models
    print("\n" + "="*60)
    rank_result = fit_rank_based_model(data)
    
    print("\n" + "="*60) 
    pairwise_result = fit_pairwise_model(data)
    
    # Step 3: Compare models
    print("\n" + "="*60)
    comparison = compare_models(rank_result, pairwise_result)
    
    # Step 4: Calculate prediction accuracy
    print("\n" + "="*60)
    accuracy = calculate_prediction_accuracy(data, rank_result, pairwise_result)
    
    # Step 5: Create visualization
    print("\n" + "="*60)
    create_model_comparison_plot(comparison)
    
    # Step 6: Save results
    if save_results:
        print("\n" + "="*60)
        print("=== SAVING RESULTS ===")
        
        # Create results summary
        results_summary = {
            'total_trials': len(data),
            'n_participants': data['userId'].nunique(),
            'rank_based_aic': comparison['rank_aic'],
            'pairwise_aic': comparison['pairwise_aic'],
            'delta_aic': comparison['delta_aic'],
            'better_model': comparison['better_model'],
            'evidence_strength': comparison['evidence_strength'],
            'rank_accuracy': accuracy['rank_accuracy'],
            'pairwise_accuracy': accuracy['pairwise_accuracy'],
            'rank_params': rank_result['params'],
            'pairwise_params': pairwise_result['params']
        }
        
        # Save as CSV
        results_df = pd.DataFrame([results_summary])
        results_df.to_csv('model_comparison_summary.csv', index=False)
        
        # Save detailed results as JSON-like format
        detailed_results = {
            'comparison': comparison,
            'accuracy': accuracy,
            'data_info': {
                'n_trials': len(data),
                'n_participants': data['userId'].nunique(),
                'response_distribution': data['response'].value_counts().to_dict()
            }
        }
        
        print("Results saved to:")
        print("  - model_comparison_summary.csv")
        print("  - model_comparison_results.png")
    
    # Final summary
    print("\n" + "="*60)
    print("=== ANALYSIS COMPLETE ===")
    print(f"Winner: {comparison['better_model'].upper()} model")
    print(f"Evidence: {comparison['evidence_strength']}")
    print(f"Delta AIC: {comparison['delta_aic']:.2f}")
    print("="*60)
    
    return {
        'data': data,
        'rank_result': rank_result,
        'pairwise_result': pairwise_result,
        'comparison': comparison,
        'accuracy': accuracy
    }

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage with your data files
    stimuli_file = "stimuli_multinum_25May25.csv"
    response_file = "filtered_data_vpsyco_numerical.csv"
    
    # Run complete analysis
    results = run_complete_analysis(stimuli_file, response_file)
    
    if results:
        print("\nAnalysis completed successfully!")
        print("Check the generated files and plots for detailed results.")
    else:
        print("\nAnalysis failed. Check your data files and try again.")