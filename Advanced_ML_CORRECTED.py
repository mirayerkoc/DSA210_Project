#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
ADVANCED ML - FORECASTING WITH NO2_lag (CORRECTED)
Traffic and Air Quality - Istanbul D-100 Highway
=============================================================================

PURPOSE: NO2 FORECASTING (not causal analysis)
This script uses NO2_lag for OPERATIONAL FORECASTING where past pollution
measurements are available and the goal is accurate short-term prediction.

CRITICAL CORRECTIONS:
1. TimeSeriesSplit used for GridSearchCV (no future data leakage)
2. Reduced feature set (removed redundant features)
3. Clear documentation of autocorrelation dominance
4. Comparison with NO2_lag vs without NO2_lag

IMPORTANT: Compare with ML_analysis_CORRECTED.py to see the difference
between causal analysis (without lag) and forecasting (with lag).

Dataset: MASTER_enriched_data.csv
Author: Miray Erkoc (30815)
Course: DSA210 - Introduction to Data Science

=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

import joblib

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100
sns.set_style("whitegrid")

print("="*80)
print("ADVANCED ML: FORECASTING WITH AUTOCORRELATION (CORRECTED)")
print("="*80)
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nCORRECTIONS APPLIED:")
print(f"  ✓ TimeSeriesSplit in GridSearchCV")
print(f"  ✓ Reduced multicollinearity")
print(f"  ✓ Clear forecasting vs causal distinction\n")

# =============================================================================
# DATA LOADING
# =============================================================================

desktop = os.path.join(os.path.expanduser("~"), "Desktop")
csv_path = os.path.join(desktop, "MASTER_enriched_data.csv")
advanced_ml_dir = os.path.join(desktop, "advanced_ml_results_corrected")
os.makedirs(advanced_ml_dir, exist_ok=True)

print(f"Loading dataset...")
df = pd.read_csv(csv_path)
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)
print(f"Loaded: {len(df):,} observations\n")

# =============================================================================
# FEATURE ENGINEERING - REDUCED SET
# =============================================================================

print("="*80)
print("FEATURE ENGINEERING - OPTIMIZED FOR FORECASTING")
print("="*80)

# Base features (reduced multicollinearity)
base_features = [
    'NUMBER_OF_VEHICLES',
    'AVERAGE_SPEED',
    'traffic_density',
    'hour',
    'dayofweek',
    'month',
    'is_weekend',
    'is_special_day'
]

# Lag features (INCLUDES NO2_lag for forecasting)
lag_features = []
for lag in [1, 2]:  # Reduced from 3 to 2 (less multicollinearity)
    for var in ['vehicles', 'speed', 'no2']:
        col = f'{var}_lag{lag}'
        if col in df.columns:
            lag_features.append(col)

# Interaction features (selective, avoid redundancy)
df['vehicle_speed_interaction'] = df['NUMBER_OF_VEHICLES'] * df['AVERAGE_SPEED']
df['hour_vehicle_interaction'] = df['hour'] * df['NUMBER_OF_VEHICLES']
interaction_features = ['vehicle_speed_interaction', 'hour_vehicle_interaction']

# Cyclical features
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
cyclical_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos']

# Combined features
all_features = base_features + lag_features + interaction_features + cyclical_features
available_features = [f for f in all_features if f in df.columns]

print(f"\nFeature Summary:")
print(f"  Base: {len(base_features)}")
print(f"  Lag (includes NO2_lag1/2): {len(lag_features)}")
print(f"  Interaction: {len(interaction_features)}")
print(f"  Cyclical: {len(cyclical_features)}")
print(f"  TOTAL: {len(available_features)}")

# Check NO2_lag presence
no2_lags_present = [f for f in available_features if 'no2_lag' in f]
print(f"\n⚠️  NO2_lag features: {no2_lags_present}")
print(f"   PURPOSE: Short-term forecasting (1-2 hours ahead)")
print(f"   NOTE: These will dominate feature importance (expected!)")

# =============================================================================
# DATA PREPARATION
# =============================================================================

print(f"\n{'='*80}")
print("DATA PREPARATION")
print(f"{'='*80}")

target = 'Concentration_NO2'
df_clean = df[available_features + [target]].dropna()

print(f"\nDataset: {len(df_clean):,} observations")
print(f"Features: {len(available_features)}")

X = df_clean[available_features]
y = df_clean[target]

# CORRECTED: Time-based split (no shuffle)
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print(f"\nTime-based split:")
print(f"  Training: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Test: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")

# =============================================================================
# BASELINE: MODEL WITHOUT NO2_lag
# =============================================================================

print(f"\n{'='*80}")
print("BASELINE: XGBoost WITHOUT NO2_lag (for comparison)")
print(f"{'='*80}")

# Create feature set without NO2_lag
features_no_lag = [f for f in available_features if 'no2_lag' not in f]
X_train_no_lag = X_train[features_no_lag]
X_test_no_lag = X_test[features_no_lag]

# Train simple XGBoost
baseline_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    verbosity=0
)

baseline_model.fit(X_train_no_lag, y_train)
baseline_pred = baseline_model.predict(X_test_no_lag)

baseline_r2 = r2_score(y_test, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

print(f"\nBaseline (WITHOUT NO2_lag):")
print(f"  R²: {baseline_r2:.4f}")
print(f"  RMSE: {baseline_rmse:.2f} µg/m³")

# =============================================================================
# XGBOOST WITH NO2_lag + TIMESERIESPLIT
# =============================================================================

print(f"\n{'='*80}")
print("XGBOOST WITH NO2_lag + TimeSeriesSplit GridSearchCV")
print(f"{'='*80}")

# CORRECTED: Use TimeSeriesSplit for CV
tscv = TimeSeriesSplit(n_splits=5)

# Reduced parameter grid (faster, less overfitting risk)
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200],
    'min_child_weight': [1, 3]
}

total_fits = np.prod([len(v) for v in param_grid.values()]) * 5
print(f"\nParameter grid: {total_fits} total fits")
print(f"  Using TimeSeriesSplit (n_splits=5)")
print(f"\nThis may take 5-10 minutes...\n")

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    verbosity=0
)

# GridSearchCV with TimeSeriesSplit
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=tscv,  # CORRECTED: TimeSeriesSplit instead of default
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\n{'='*80}")
print("HYPERPARAMETER TUNING RESULTS")
print(f"{'='*80}")
print(f"\nBest parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

best_xgb = grid_search.best_estimator_

y_pred_train = best_xgb.predict(X_train)
y_pred_test = best_xgb.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\nFinal XGBoost Performance (WITH NO2_lag):")
print(f"  Training R²: {train_r2:.4f}")
print(f"  Test R²: {test_r2:.4f}")
print(f"  Test RMSE: {test_rmse:.2f} µg/m³")

# =============================================================================
# LIGHTGBM COMPARISON
# =============================================================================

if LIGHTGBM_AVAILABLE:
    print(f"\n{'='*80}")
    print("LIGHTGBM COMPARISON")
    print(f"{'='*80}")
    
    lgb_model = lgb.LGBMRegressor(
        **grid_search.best_params_,
        random_state=42,
        verbose=-1
    )
    
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)
    
    lgb_r2 = r2_score(y_test, lgb_pred)
    lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_pred))
    
    print(f"\nLightGBM Performance:")
    print(f"  Test R²: {lgb_r2:.4f}")
    print(f"  Test RMSE: {lgb_rmse:.2f} µg/m³")
    
    comparison = pd.DataFrame({
        'Model': ['Baseline (no NO2_lag)', 'XGBoost (with NO2_lag)', 'LightGBM (with NO2_lag)'],
        'R2': [baseline_r2, test_r2, lgb_r2],
        'RMSE': [baseline_rmse, test_rmse, lgb_rmse]
    })
    
    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}")
    print(comparison.to_string(index=False))
    
    improvement = ((test_r2 - baseline_r2) / baseline_r2) * 100
    print(f"\nImprovement with NO2_lag: {improvement:.1f}%")
    print(f"This demonstrates NO2's strong autocorrelation")
    
    comparison.to_csv(os.path.join(advanced_ml_dir, 'model_comparison_corrected.csv'), index=False)

# =============================================================================
# FEATURE IMPORTANCE ANALYSIS
# =============================================================================

print(f"\n{'='*80}")
print("FEATURE IMPORTANCE - AUTOCORRELATION DOMINANCE")
print(f"{'='*80}")

importances = best_xgb.feature_importances_
importance_df = pd.DataFrame({
    'Feature': available_features,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(f"\nTop 15 Features:")
print(importance_df.head(15).to_string(index=False))

# Check NO2_lag dominance
no2_lag_importance = importance_df[importance_df['Feature'].str.contains('no2_lag')]['Importance'].sum()
print(f"\n⚠️  NO2_lag features combined importance: {no2_lag_importance*100:.1f}%")
print(f"   This is EXPECTED for forecasting models")
print(f"   Traffic features have {((1-no2_lag_importance)*100):.1f}% importance")

# Visualization
fig, ax = plt.subplots(figsize=(10, 12))
top_features = importance_df.head(20)
colors = ['red' if 'no2_lag' in f else 'steelblue' for f in top_features['Feature']]
ax.barh(range(len(top_features)), top_features['Importance'], color=colors)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'])
ax.invert_yaxis()
ax.set_xlabel('Importance', fontweight='bold')
ax.set_title('Feature Importance (Red = NO2_lag, Blue = Others)', 
             fontweight='bold', fontsize=14)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(advanced_ml_dir, 'feature_importance_corrected.png'), 
            dpi=300, bbox_inches='tight')
plt.close()
print(f"\n  Saved: feature_importance_corrected.png")

# =============================================================================
# LEARNING CURVES
# =============================================================================

print(f"\n{'='*80}")
print("LEARNING CURVES")
print(f"{'='*80}")

train_sizes = np.linspace(0.1, 1.0, 10)
train_scores = []
test_scores = []

print(f"\nCalculating learning curves...")
for size in train_sizes:
    n_samples = int(len(X_train) * size)
    X_subset = X_train[:n_samples]
    y_subset = y_train[:n_samples]
    
    temp_model = xgb.XGBRegressor(**grid_search.best_params_, verbosity=0)
    temp_model.fit(X_subset, y_subset)
    
    train_score = r2_score(y_subset, temp_model.predict(X_subset))
    test_score = r2_score(y_test, temp_model.predict(X_test))
    
    train_scores.append(train_score)
    test_scores.append(test_score)
    
    print(f"  {size*100:3.0f}%: Train R²={train_score:.4f}, Test R²={test_score:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_sizes * 100, train_scores, 'o-', label='Training', linewidth=2, markersize=8)
ax.plot(train_sizes * 100, test_scores, 's-', label='Test', linewidth=2, markersize=8)
ax.set_xlabel('Training Set Size (%)', fontweight='bold')
ax.set_ylabel('R² Score', fontweight='bold')
ax.set_title('Learning Curves - XGBoost with NO2_lag', fontweight='bold', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(advanced_ml_dir, 'learning_curves_corrected.png'), 
            dpi=300, bbox_inches='tight')
plt.close()
print(f"\n  Saved: learning_curves_corrected.png")

# =============================================================================
# RESIDUAL ANALYSIS
# =============================================================================

print(f"\n{'='*80}")
print("RESIDUAL ANALYSIS")
print(f"{'='*80}")

residuals = y_test.values - y_pred_test

print(f"\nResidual Statistics:")
print(f"  Mean: {residuals.mean():.2f}")
print(f"  Std: {residuals.std():.2f}")
print(f"  Min: {residuals.min():.2f}")
print(f"  Max: {residuals.max():.2f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Predictions vs Actual
axes[0, 0].scatter(y_test, y_pred_test, alpha=0.5, s=20)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect')
axes[0, 0].set_xlabel('Actual NO2', fontweight='bold')
axes[0, 0].set_ylabel('Predicted NO2', fontweight='bold')
axes[0, 0].set_title('Predictions vs Actual', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Residual plot
axes[0, 1].scatter(y_pred_test, residuals, alpha=0.5, s=20, color='orange')
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted NO2', fontweight='bold')
axes[0, 1].set_ylabel('Residuals', fontweight='bold')
axes[0, 1].set_title('Residual Plot', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Histogram
axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Residuals', fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontweight='bold')
axes[1, 0].set_title('Residual Distribution', fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)

# Q-Q plot
from scipy import stats as sp_stats
sp_stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(advanced_ml_dir, 'residual_analysis_corrected.png'), 
            dpi=300, bbox_inches='tight')
plt.close()
print(f"\n  Saved: residual_analysis_corrected.png")

# =============================================================================
# SAVE MODEL
# =============================================================================

print(f"\n{'='*80}")
print("SAVING MODEL")
print(f"{'='*80}")

model_path = os.path.join(advanced_ml_dir, 'xgboost_forecasting_model.pkl')
joblib.dump(best_xgb, model_path)
print(f"\nSaved: xgboost_forecasting_model.pkl")

feature_path = os.path.join(advanced_ml_dir, 'feature_names.txt')
with open(feature_path, 'w') as f:
    f.write('\n'.join(available_features))
print(f"Saved: feature_names.txt")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(f"\n{'='*80}")
print("CORRECTED ADVANCED ML - FINAL SUMMARY")
print(f"{'='*80}")

print(f"\nCORRECTIONS APPLIED:")
print(f"  1. ✓ TimeSeriesSplit in GridSearchCV (no future leakage)")
print(f"  2. ✓ Reduced features (less multicollinearity)")
print(f"  3. ✓ Baseline comparison (with vs without NO2_lag)")

print(f"\nKEY RESULTS:")
print(f"  • Baseline (no NO2_lag): R² = {baseline_r2:.4f}")
print(f"  • XGBoost (with NO2_lag): R² = {test_r2:.4f}")
print(f"  • Improvement: {((test_r2-baseline_r2)/baseline_r2*100):.1f}%")
print(f"  • NO2_lag importance: {no2_lag_importance*100:.1f}%")

print(f"\nINTERPRETATION:")
print(f"  • High R² is due to AUTOCORRELATION (NO2_lag dominates)")
print(f"  • This is APPROPRIATE for forecasting applications")
print(f"  • For CAUSAL analysis, see ML_analysis_CORRECTED.py")

print(f"\nUSE CASES:")
print(f"  • This model: 1-2 hour NO2 forecasting")
print(f"  • ML_analysis_CORRECTED: Policy impact assessment")

print(f"\nOutput: {advanced_ml_dir}")
print(f"{'='*80}")
print("CORRECTED ANALYSIS COMPLETE")
print(f"{'='*80}")
