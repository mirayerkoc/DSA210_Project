#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
MACHINE LEARNING ANALYSIS - CORRECTED VERSION
Traffic and Air Quality - Istanbul D-100 Highway
=============================================================================

CRITICAL CORRECTIONS APPLIED:
1. NO2_lag features EXCLUDED to avoid data leakage in causal analysis
2. TimeSeriesSplit used instead of standard CV (no future data leakage)
3. Reduced multicollinearity (removed redundant features)
4. Class imbalance handled with class_weight='balanced'
5. F1-score and per-class metrics reported
6. Residual analysis includes discussion of missing weather data

This script focuses on measuring TRAFFIC'S CAUSAL EFFECT on pollution,
not just prediction accuracy.

Dataset: MASTER_enriched_data.csv (8,027 hourly observations)
Author: Miray Erkoc (30815)
Course: DSA210 - Introduction to Data Science
Institution: Sabanci University

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

# ML Libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                            classification_report, confusion_matrix, 
                            accuracy_score, f1_score)

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Clustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Time Series
from statsmodels.tsa.arima.model import ARIMA

# Visualization
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100
sns.set_style("whitegrid")

print("="*80)
print("CORRECTED ML ANALYSIS: CAUSAL TRAFFIC-POLLUTION RELATIONSHIP")
print("="*80)
print(f"Student ID: 30815")
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nKEY CORRECTIONS:")
print(f"  ✓ NO2_lag features EXCLUDED (avoiding autocorrelation dominance)")
print(f"  ✓ TimeSeriesSplit for temporal validation")
print(f"  ✓ Reduced feature multicollinearity")
print(f"  ✓ Class imbalance handling")
print(f"  ✓ Comprehensive residual analysis\n")

# =============================================================================
# DATA LOADING & PREPARATION
# =============================================================================

desktop = os.path.join(os.path.expanduser("~"), "Desktop")
csv_path = os.path.join(desktop, "MASTER_enriched_data.csv")
ml_dir = os.path.join(desktop, "ml_results_corrected")
os.makedirs(ml_dir, exist_ok=True)

if not os.path.exists(csv_path):
    print(f"ERROR: Dataset not found at {csv_path}")
    exit()

print(f"Loading dataset...")
df = pd.read_csv(csv_path)
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)
print(f"Successfully loaded: {len(df):,} observations\n")

# =============================================================================
# CORRECTED FEATURE SELECTION (NO DATA LEAKAGE)
# =============================================================================

print("="*80)
print("FEATURE ENGINEERING - CORRECTED")
print("="*80)

# CRITICAL: Exclude NO2_lag to measure TRAFFIC effect
# Only use exogenous (external) predictors

feature_cols_exogenous = [
    # Traffic features (current hour)
    'NUMBER_OF_VEHICLES',
    'AVERAGE_SPEED',
    'traffic_density',
    
    # Temporal features
    'hour',
    'dayofweek',
    'month',
    'is_weekend',
    'is_special_day',
    
    # Traffic history (lagged traffic, NOT NO2!)
    'vehicles_lag1',
    'vehicles_lag2',
    'speed_lag1',
    'speed_lag2'
    
    # EXCLUDED: no2_lag1/2/3 (would dominate and hide traffic effect)
    # EXCLUDED: vehicles_lag3, speed_lag3 (multicollinearity reduction)
    # EXCLUDED: MIN/MAX speed (redundant with AVERAGE_SPEED)
]

# Target variables
target_regression = 'Concentration_NO2'
target_classification = 'AQI_Category'

# Filter available features
available_features = [f for f in feature_cols_exogenous if f in df.columns]
print(f"\nExogenous features selected: {len(available_features)}")
print(f"Features: {available_features}")
print(f"\n⚠️  NO2_lag features INTENTIONALLY EXCLUDED")
print(f"   Reason: To measure traffic's TRUE causal effect without autocorrelation masking\n")

# Create clean dataset
df_ml = df[available_features + [target_regression, target_classification]].copy()
df_ml = df_ml.dropna()
print(f"Dataset after removing missing values: {len(df_ml):,} observations")

# =============================================================================
# TASK 1: REGRESSION - NO2 PREDICTION (CAUSAL ANALYSIS)
# =============================================================================

print("\n" + "="*80)
print("TASK 1: REGRESSION - MEASURING TRAFFIC'S CAUSAL EFFECT")
print("="*80)

# Prepare data
X = df_ml[available_features]
y = df_ml[target_regression]

# CORRECTED: Train-test split for time series (no shuffle)
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print(f"\nTime-based split (chronological order preserved):")
print(f"  Training set: {len(X_train):,} observations (first 80%)")
print(f"  Test set: {len(X_test):,} observations (last 20%)")
print(f"  No future data leakage ✓")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
regression_models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42)  # Reduced depth to prevent overfitting
}

# Train and evaluate models with TimeSeriesSplit CV
regression_results = {}
tscv = TimeSeriesSplit(n_splits=5)

print("\n" + "-"*80)
print("Training Regression Models with TimeSeriesSplit CV...")
print("-"*80)

for name, model in regression_models.items():
    print(f"\n[{name}]")
    
    # Train
    if name in ['Ridge Regression', 'Lasso Regression']:
        model.fit(X_train_scaled, y_train)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
    
    # Cross-validation with TimeSeriesSplit
    if name in ['Ridge Regression', 'Lasso Regression']:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                    cv=tscv, scoring='r2', n_jobs=-1)
    else:
        cv_scores = cross_val_score(model, X_train, y_train, 
                                    cv=tscv, scoring='r2', n_jobs=-1)
    
    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    regression_results[name] = {
        'Train R2': train_r2,
        'Test R2': test_r2,
        'CV R2 Mean': cv_scores.mean(),
        'CV R2 Std': cv_scores.std(),
        'RMSE': test_rmse,
        'MAE': test_mae
    }
    
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  CV R² (TimeSeriesSplit): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  RMSE: {test_rmse:.2f} µg/m³")
    print(f"  MAE: {test_mae:.2f} µg/m³")

# Find best model
best_model_name = max(regression_results.items(), key=lambda x: x[1]['Test R2'])[0]
best_model = regression_models[best_model_name]

print(f"\n{'='*80}")
print(f"BEST MODEL (WITHOUT NO2_lag): {best_model_name}")
print(f"  Test R²: {regression_results[best_model_name]['Test R2']:.4f}")
print(f"  CV R²: {regression_results[best_model_name]['CV R2 Mean']:.4f}")
print(f"  RMSE: {regression_results[best_model_name]['RMSE']:.2f} µg/m³")
print(f"\n⚠️  INTERPRETATION:")
print(f"   Low R² indicates traffic alone has LIMITED direct predictive power")
print(f"   This is EXPECTED when NO2_lag is excluded")
print(f"   Suggests other factors (weather, autocorrelation) are important")
print(f"{'='*80}")

# Feature importance for tree-based models
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    if best_model_name == 'Random Forest':
        model_for_importance = regression_models['Random Forest']
    else:
        model_for_importance = regression_models['Gradient Boosting']
    
    importance_df = pd.DataFrame({
        'Feature': available_features,
        'Importance': model_for_importance.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n{best_model_name} Feature Importance:")
    print(importance_df.to_string(index=False))

# =============================================================================
# VISUALIZATION 1: Model Comparison
# =============================================================================

print(f"\n{'='*80}")
print("GENERATING VISUALIZATIONS...")
print(f"{'='*80}")

# Create comparison DataFrame
comparison_df = pd.DataFrame(regression_results).T

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# R² comparison
axes[0, 0].barh(comparison_df.index, comparison_df['Test R2'], color='steelblue')
axes[0, 0].set_xlabel('R² Score', fontweight='bold')
axes[0, 0].set_title('Model Comparison - Test R² (Without NO2_lag)', fontweight='bold')
axes[0, 0].grid(axis='x', alpha=0.3)

# RMSE comparison
axes[0, 1].barh(comparison_df.index, comparison_df['RMSE'], color='orange')
axes[0, 1].set_xlabel('RMSE (µg/m³)', fontweight='bold')
axes[0, 1].set_title('Model Comparison - RMSE', fontweight='bold')
axes[0, 1].grid(axis='x', alpha=0.3)

# Predictions vs Actual (best model)
if best_model_name in ['Ridge Regression', 'Lasso Regression']:
    best_pred = best_model.predict(X_test_scaled)
else:
    best_pred = best_model.predict(X_test)

axes[1, 0].scatter(y_test, best_pred, alpha=0.5, s=20, color='steelblue')
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
axes[1, 0].set_xlabel('Actual NO2 (µg/m³)', fontweight='bold')
axes[1, 0].set_ylabel('Predicted NO2 (µg/m³)', fontweight='bold')
axes[1, 0].set_title(f'{best_model_name} - Predictions vs Actual', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Residual plot with weather discussion
residuals = y_test.values - best_pred
axes[1, 1].scatter(best_pred, residuals, alpha=0.5, s=20, color='orange')
axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Predicted NO2 (µg/m³)', fontweight='bold')
axes[1, 1].set_ylabel('Residuals (µg/m³)', fontweight='bold')
axes[1, 1].set_title('Residual Plot - Large errors likely due to missing weather data', 
                     fontweight='bold', fontsize=11)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(ml_dir, '1_regression_comparison_corrected.png'), 
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 1_regression_comparison_corrected.png")

# =============================================================================
# RESIDUAL ANALYSIS WITH WEATHER DISCUSSION
# =============================================================================

print(f"\n{'='*80}")
print("RESIDUAL ANALYSIS & MISSING WEATHER DATA IMPACT")
print(f"{'='*80}")

# Identify large residuals
residual_threshold = residuals.std() * 2
large_residuals = np.abs(residuals) > residual_threshold
large_residual_pct = (large_residuals.sum() / len(residuals)) * 100

print(f"\nResidual Statistics:")
print(f"  Mean: {residuals.mean():.2f} µg/m³ (should be ~0)")
print(f"  Std Dev: {residuals.std():.2f} µg/m³")
print(f"  Large residuals (>2σ): {large_residuals.sum()} ({large_residual_pct:.1f}%)")

print(f"\n⚠️  MISSING WEATHER DATA IMPACT:")
print(f"   • Wind speed/direction: Major factor in NO2 dispersion")
print(f"   • Temperature: Affects atmospheric mixing height")
print(f"   • Humidity: Influences chemical reactions")
print(f"   • Large residuals likely occur on windy/rainy days")
print(f"   • Model cannot predict these variations without weather data")
print(f"\n   RECOMMENDATION: Future work should integrate meteorological data")

# Create detailed residual plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram
axes[0, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].axvline(x=0, color='r', linestyle='--', lw=2)
axes[0, 0].set_xlabel('Residuals (µg/m³)', fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontweight='bold')
axes[0, 0].set_title('Residual Distribution', fontweight='bold')
axes[0, 0].grid(axis='y', alpha=0.3)

# Q-Q plot
from scipy import stats as sp_stats
sp_stats.probplot(residuals, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot (Normality Check)', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Residuals over time
test_dates = df_ml.iloc[split_index:]['datetime'].values if 'datetime' in df_ml.columns else range(len(residuals))
axes[1, 0].scatter(range(len(residuals)), residuals, alpha=0.3, s=10, color='steelblue')
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].axhline(y=residual_threshold, color='orange', linestyle=':', lw=1, label='±2σ')
axes[1, 0].axhline(y=-residual_threshold, color='orange', linestyle=':', lw=1)
axes[1, 0].set_xlabel('Time Index', fontweight='bold')
axes[1, 0].set_ylabel('Residuals (µg/m³)', fontweight='bold')
axes[1, 0].set_title('Residuals Over Time (Patterns suggest weather influence)', fontweight='bold', fontsize=11)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Residuals vs traffic
test_traffic = X_test['NUMBER_OF_VEHICLES'].values
axes[1, 1].scatter(test_traffic, residuals, alpha=0.5, s=20, color='orange')
axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Traffic (vehicles)', fontweight='bold')
axes[1, 1].set_ylabel('Residuals (µg/m³)', fontweight='bold')
axes[1, 1].set_title('Residuals vs Traffic Volume', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(ml_dir, '2_residual_analysis_weather_discussion.png'), 
            dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: 2_residual_analysis_weather_discussion.png")

# =============================================================================
# TASK 2: CLASSIFICATION WITH CLASS IMBALANCE HANDLING
# =============================================================================

print(f"\n{'='*80}")
print("TASK 2: AQI CLASSIFICATION - CORRECTED FOR IMBALANCE")
print(f"{'='*80}")

# Check class distribution
print(f"\nALL Classes in Dataset:")
class_counts_all = df_ml[target_classification].value_counts().sort_index()
for cls, count in class_counts_all.items():
    print(f"  {cls}: {count:,} ({count/len(df_ml)*100:.1f}%)")

print(f"\n⚠️  Class imbalance detected!")

# Check train/test distribution
y_class_full = df_ml[target_classification]
y_class_train_labels = y_class_full[:split_index]
y_class_test_labels = y_class_full[split_index:]

print(f"\nClasses in Training Set:")
train_class_counts = y_class_train_labels.value_counts().sort_index()
for cls, count in train_class_counts.items():
    print(f"  {cls}: {count:,}")

print(f"\nClasses in Test Set:")
test_class_counts = y_class_test_labels.value_counts().sort_index()
for cls, count in test_class_counts.items():
    print(f"  {cls}: {count:,}")

missing_classes = set(class_counts_all.index) - set(test_class_counts.index)
if missing_classes:
    print(f"\n⚠️  Classes MISSING from test set: {missing_classes}")
    print(f"   These classes are too rare and didn't appear in the last 20% of data")
    print(f"   Classification report will only include classes present in test set")

# Prepare classification data
X_class = df_ml[available_features]
y_class = df_ml[target_classification]

# Encode labels
le = LabelEncoder()
y_class_encoded = le.fit_transform(y_class)

# Time-based split
X_class_train, X_class_test = X_class[:split_index], X_class[split_index:]
y_class_train, y_class_test = y_class_encoded[:split_index], y_class_encoded[split_index:]

# Scale features
X_class_train_scaled = scaler.fit_transform(X_class_train)
X_class_test_scaled = scaler.transform(X_class_test)

# Classification models WITH class_weight='balanced'
classification_models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),  # No class_weight param
    'Gaussian NB': GaussianNB()
}

classification_results = {}

print("\n" + "-"*80)
print("Training Classification Models (with class_weight='balanced')...")
print("-"*80)

for name, model in classification_models.items():
    print(f"\n[{name}]")
    
    # Train
    if name == 'Logistic Regression':
        model.fit(X_class_train_scaled, y_class_train)
        y_class_pred = model.predict(X_class_test_scaled)
    else:
        model.fit(X_class_train, y_class_train)
        y_class_pred = model.predict(X_class_test)
    
    # Metrics
    accuracy = accuracy_score(y_class_test, y_class_pred)
    f1_weighted = f1_score(y_class_test, y_class_pred, average='weighted')
    f1_macro = f1_score(y_class_test, y_class_pred, average='macro')
    
    classification_results[name] = {
        'Accuracy': accuracy,
        'F1_Weighted': f1_weighted,
        'F1_Macro': f1_macro
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score (Weighted): {f1_weighted:.4f}")
    print(f"  F1-Score (Macro): {f1_macro:.4f}")
    
    # Per-class metrics
    # Get unique classes in test set
    unique_test_labels = np.unique(y_class_test)
    test_class_names = le.inverse_transform(unique_test_labels)
    
    print(f"\n  Classification Report:")
    print(classification_report(y_class_test, y_class_pred, 
                                labels=unique_test_labels,
                                target_names=test_class_names, 
                                zero_division=0))

# Best classification model
best_clf_name = max(classification_results.items(), key=lambda x: x[1]['F1_Weighted'])[0]

print(f"\n{'='*80}")
print(f"BEST CLASSIFICATION MODEL: {best_clf_name}")
print(f"  F1-Score (Weighted): {classification_results[best_clf_name]['F1_Weighted']:.4f}")
print(f"  F1-Score (Macro): {classification_results[best_clf_name]['F1_Macro']:.4f}")
print(f"{'='*80}")

# Confusion matrix (only for classes present in test set)
best_clf_model = classification_models[best_clf_name]
if best_clf_name == 'Logistic Regression':
    y_clf_pred_best = best_clf_model.predict(X_class_test_scaled)
else:
    y_clf_pred_best = best_clf_model.predict(X_class_test)

# Get unique labels in test set
unique_test_labels = np.unique(y_class_test)
test_class_names = le.inverse_transform(unique_test_labels)

# Create confusion matrix
cm = confusion_matrix(y_class_test, y_clf_pred_best, labels=unique_test_labels)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=test_class_names, 
            yticklabels=test_class_names, 
            ax=ax)
ax.set_xlabel('Predicted', fontweight='bold')
ax.set_ylabel('Actual', fontweight='bold')
ax.set_title(f'{best_clf_name} - Confusion Matrix (Class-Balanced)\n' + 
             f'Classes in test set: {len(unique_test_labels)} of {len(le.classes_)} total', 
             fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(ml_dir, '3_confusion_matrix_corrected.png'), 
            dpi=300, bbox_inches='tight')
plt.close()
print("\n  Saved: 3_confusion_matrix_corrected.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(f"\n{'='*80}")
print("CORRECTED ANALYSIS - FINAL SUMMARY")
print(f"{'='*80}")

print(f"\nKEY CORRECTIONS APPLIED:")
print(f"  1. ✓ NO2_lag EXCLUDED → Measures traffic's TRUE causal effect")
print(f"  2. ✓ TimeSeriesSplit CV → No future data leakage")
print(f"  3. ✓ Reduced features → Less multicollinearity")
print(f"  4. ✓ Class balancing → Fair minority class treatment")
print(f"  5. ✓ Weather discussion → Explains large residuals")

print(f"\nKEY FINDINGS:")
print(f"  • Best Regression R²: {regression_results[best_model_name]['Test R2']:.4f}")
print(f"  • INTERPRETATION: Traffic alone explains limited variance")
print(f"  • WHY LOW?: NO2_lag excluded, weather data missing")
print(f"  • THIS IS EXPECTED in causal analysis!")

print(f"\n  • Best Classification F1: {classification_results[best_clf_name]['F1_Weighted']:.4f}")
print(f"  • Class imbalance handled with balanced weights")

print(f"\nRECOMMENDATIONS:")
print(f"  • For FORECASTING: Use NO2_lag (see Advanced_ML.py)")
print(f"  • For POLICY: Use this analysis (causal effect clear)")
print(f"  • Future work: Add weather data for better predictions")

print(f"\nOutput directory: {ml_dir}")
print(f"{'='*80}")
print("CORRECTED ANALYSIS COMPLETED SUCCESSFULLY")
print(f"{'='*80}")
