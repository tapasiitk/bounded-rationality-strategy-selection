import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split, KFold
import warnings
warnings.filterwarnings('ignore')

# --------------------
# STEP 1: CROSS-VALIDATION FRAMEWORK
# --------------------

def cross_validate_models(data, n_folds=5):
    """Cross-validate both models to check for overfitting"""
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {'rank': [], 'meta': []}
    
    for train_idx, test_idx in kfold.split(data):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        
        # Fit models on training data
        rank_fit = fit_rank(train_data)
        meta_fit = fit_meta(train_data)
        
        # Evaluate on test data
        rank_test_nll = negloglik_rank(rank_fit.x, test_data)
        meta_test_nll = negloglik_meta(meta_fit.x, test_data)
        
        results['rank'].append(rank_test_nll)
        results['meta'].append(meta_test_nll)
        
        print(f"Fold: Rank test NLL = {rank_test_nll:.2f}, Meta test NLL = {meta_test_nll:.2f}")
    
    return results

# --------------------
# STEP 2: PREDICTION ACCURACY
# --------------------

def calculate_prediction_accuracy(data, params_rank, params_meta, n_test=200):
    """Calculate actual prediction accuracy on held-out data"""
    
    # Hold out test data
    train_data, test_data = train_test_split(data, test_size=min(n_test, len(data)//5), random_state=42)
    
    correct_rank = 0
    correct_meta = 0
    idx = {'A': 0, 'B': 1, 'C': 2}
    rev_idx = {0: 'A', 1: 'B', 2: 'C'}
    
    for _, row in test_data.iterrows():
        actual = row['response']
        
        # Rank-based prediction
        p_rank = rank_based_model_prediction(row['A_frames'], row['B_frames'], row['C_frames'], *params_rank)
        pred_rank = rev_idx[np.argmax(p_rank)]
        if pred_rank == actual:
            correct_rank += 1
            
        # Meta-model prediction  
        p_meta = meta_model(row['A_frames'], row['B_frames'], row['C_frames'], *params_meta)
        pred_meta = rev_idx[np.argmax(p_meta)]
        if pred_meta == actual:
            correct_meta += 1
    
    n = len(test_data)
    print(f"Prediction Accuracy (n={n}):")
    print(f"  Rank-based: {correct_rank}/{n} ({100*correct_rank/n:.1f}%)")
    print(f"  Meta-model: {correct_meta}/{n} ({100*correct_meta/n:.1f}%)")
    
    return correct_rank/n, correct_meta/n

# --------------------
# STEP 3: PARAMETER RECOVERY TEST
# --------------------

def parameter_recovery_test(data_sample, n_recovery=5):
    """Test if models can recover known parameters from simulated data"""
    
    print("=== PARAMETER RECOVERY TEST ===")
    
    # Test rank-based model recovery
    true_params_rank = [1.2, 0.6, 0.2, 15.0]
    recovery_errors_rank = []
    
    for i in range(n_recovery):
        # Simulate data with known parameters
        sim_responses = []
        idx = {0: 'A', 1: 'B', 2: 'C'}
        
        for _, row in data_sample.iterrows():
            p = rank_based_model_prediction(row['A_frames'], row['B_frames'], row['C_frames'], *true_params_rank)
            choice = np.random.choice(3, p=p)
            sim_responses.append(idx[choice])
        
        sim_data = data_sample.copy()
        sim_data['response'] = sim_responses
        
        # Fit model to simulated data
        fit_result = fit_rank(sim_data)
        error = np.mean(np.abs(np.array(fit_result.x) - np.array(true_params_rank)))
        recovery_errors_rank.append(error)
    
    print(f"Rank-based parameter recovery error: {np.mean(recovery_errors_rank):.3f} ± {np.std(recovery_errors_rank):.3f}")
    
    # Test meta-model recovery
    true_params_meta = [0.8, 2.0, 3.0]
    recovery_errors_meta = []
    
    for i in range(n_recovery):
        sim_responses = []
        
        for _, row in data_sample.iterrows():
            p = meta_model(row['A_frames'], row['B_frames'], row['C_frames'], *true_params_meta)
            choice = np.random.choice(3, p=p)
            sim_responses.append(idx[choice])
        
        sim_data = data_sample.copy()
        sim_data['response'] = sim_responses
        
        fit_result = fit_meta(sim_data)
        error = np.mean(np.abs(np.array(fit_result.x) - np.array(true_params_meta)))
        recovery_errors_meta.append(error)
    
    print(f"Meta-model parameter recovery error: {np.mean(recovery_errors_meta):.3f} ± {np.std(recovery_errors_meta):.3f}")

# --------------------
# STEP 4: ROBUSTNESS CHECKS
# --------------------

def robustness_check(data, n_runs=5):
    """Test model fitting with different starting values"""
    
    print("=== ROBUSTNESS CHECK ===")
    
    # Multiple random starts for rank-based model
    rank_fits = []
    for i in range(n_runs):
        initial = [np.random.uniform(0.5, 2), np.random.uniform(0.2, 1.5), np.random.uniform(0.05, 0.8), np.random.uniform(5, 50)]
        try:
            bounds = [(0.01,2),(0,2),(0,2),(1e-3,500)]
            res = minimize(negloglik_rank, initial, args=(data,), bounds=bounds)
            if res.success:
                rank_fits.append(res)
        except:
            continue
    
    # Multiple random starts for meta-model
    meta_fits = []
    for i in range(n_runs):
        initial = [np.random.uniform(0.2, 0.8), np.random.uniform(0.5, 5), np.random.uniform(0.5, 5)]
        try:
            bounds = [(0,1),(1e-3,20),(1e-3,20)]
            res = minimize(negloglik_meta, initial, args=(data,), bounds=bounds)
            if res.success:
                meta_fits.append(res)
        except:
            continue
    
    if rank_fits:
        rank_nlls = [fit.fun for fit in rank_fits]
        print(f"Rank-based NLL consistency: {np.std(rank_nlls):.3f} (lower = more robust)")
    
    if meta_fits:
        meta_nlls = [fit.fun for fit in meta_fits]  
        print(f"Meta-model NLL consistency: {np.std(meta_nlls):.3f} (lower = more robust)")

# --------------------
# STEP 5: COMPLETE VALIDATION PIPELINE
# --------------------

# [Include the original model definitions and fitting functions here]

def complete_model_validation(data):
    """Run complete validation pipeline"""
    
    print("="*60)
    print("COMPLETE MODEL VALIDATION PIPELINE")
    print("="*60)
    
    # 1. Original fits
    print("\n1. ORIGINAL MODEL FITS")
    fit_r = fit_rank(data)  
    fit_m = fit_meta(data)
    print(f"Rank fit: {fit_r.x}, NLL = {fit_r.fun:.2f}")
    print(f"Meta fit: {fit_m.x}, NLL = {fit_m.fun:.2f}")
    
    # 2. Cross-validation
    print("\n2. CROSS-VALIDATION")
    cv_results = cross_validate_models(data)
    print(f"Rank CV NLL: {np.mean(cv_results['rank']):.2f} ± {np.std(cv_results['rank']):.2f}")
    print(f"Meta CV NLL: {np.mean(cv_results['meta']):.2f} ± {np.std(cv_results['meta']):.2f}")
    
    # 3. Prediction accuracy
    print("\n3. PREDICTION ACCURACY")
    acc_rank, acc_meta = calculate_prediction_accuracy(data, fit_r.x, fit_m.x)
    
    # 4. Parameter recovery (on subset to save time)
    print("\n4. PARAMETER RECOVERY")  
    sample_data = data.sample(n=min(200, len(data)), random_state=42)
    parameter_recovery_test(sample_data)
    
    # 5. Robustness check
    print("\n5. ROBUSTNESS CHECK")
    robustness_check(data)
    
    # 6. Final comparison with all criteria
    print("\n6. FINAL COMPARISON")
    rank_aic = 2*4 + 2*fit_r.fun
    meta_aic = 2*3 + 2*fit_m.fun
    print(f"AIC: Rank = {rank_aic:.1f}, Meta = {meta_aic:.1f}")
    print(f"Cross-val: Rank = {np.mean(cv_results['rank']):.1f}, Meta = {np.mean(cv_results['meta']):.1f}")
    print(f"Accuracy: Rank = {acc_rank:.1%}, Meta = {acc_meta:.1%}")
    
    return {
        'rank_fit': fit_r,
        'meta_fit': fit_m, 
        'cv_results': cv_results,
        'accuracies': (acc_rank, acc_meta)
    }

# Run complete validation
validation_results = complete_model_validation(df)
