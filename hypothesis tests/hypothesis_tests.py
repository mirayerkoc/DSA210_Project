#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
STATISTICAL HYPOTHESIS TESTING
Traffic and Air Quality - Istanbul D-100 Highway
=============================================================================

This script performs comprehensive hypothesis testing including:
- H1: Traffic-Pollution Correlation Analysis
- H2: Time Lag Effect (2-hour hypothesis)
- H3: Holiday/Special Day Effect
- H4: Weekend Effect
- H5: Rush Hour Pollution Peaks

Statistical Methods:
- Pearson correlation coefficients
- Independent samples t-tests
- One-way ANOVA
- Cohen's d effect size calculations

Dataset: MASTER_enriched_data.csv (8,027 hourly observations)
Author: Miray Erkoc (30815)
Course: DSA210 - Introduction to Data Science
Institution: Sabanci University

=============================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STATISTICAL HYPOTHESIS TESTING: TRAFFIC & AIR QUALITY")
print("="*80)
print(f"Student ID: 30815")
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# =============================================================================
# DATA LOADING
# =============================================================================

desktop = os.path.join(os.path.expanduser("~"), "Desktop")
csv_path = os.path.join(desktop, "MASTER_enriched_data.csv")

if not os.path.exists(csv_path):
    print(f"ERROR: Dataset not found at {csv_path}")
    exit()

print(f"Loading dataset...")
df = pd.read_csv(csv_path)
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)
print(f"Successfully loaded: {len(df):,} observations, {len(df.columns)} variables\n")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def interpret_pvalue(p_value, alpha=0.05):
    """Interpret statistical significance of p-value"""
    if p_value < 0.001:
        return "Highly significant (p < 0.001)"
    elif p_value < 0.01:
        return "Very significant (p < 0.01)"
    elif p_value < alpha:
        return "Significant (p < 0.05)"
    else:
        return "Not significant (p >= 0.05)"

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = group1.mean(), group2.mean()
    var1, var2 = group1.var(), group2.var()
    
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    d = (mean1 - mean2) / pooled_std
    
    if abs(d) < 0.2:
        interpretation = "negligible"
    elif abs(d) < 0.5:
        interpretation = "small"
    elif abs(d) < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    return d, interpretation

# =============================================================================
# H1: TRAFFIC-POLLUTION CORRELATION
# =============================================================================

print("="*80)
print("H1: TRAFFIC-POLLUTION CORRELATION ANALYSIS")
print("="*80)

h1_results = {}

# Test 1: Vehicle Count ↔ NO2
if 'NUMBER_OF_VEHICLES' in df.columns and 'Concentration_NO2' in df.columns:
    temp = df[['NUMBER_OF_VEHICLES', 'Concentration_NO2']].dropna()
    r, p = stats.pearsonr(temp['NUMBER_OF_VEHICLES'], temp['Concentration_NO2'])
    print(f"\nTest 1: Vehicle Count ↔ NO2")
    print(f"  Sample size: {len(temp):,}")
    print(f"  Pearson r: {r:.4f}")
    print(f"  P-value: {p:.6f}")
    print(f"  Result: {interpret_pvalue(p)}")
    h1_results['vehicle_no2'] = {'r': r, 'p': p, 'significant': p < 0.05}

# Test 2: Vehicle Count ↔ PM10
if 'NUMBER_OF_VEHICLES' in df.columns and 'AQI_PM10' in df.columns:
    temp = df[['NUMBER_OF_VEHICLES', 'AQI_PM10']].dropna()
    r, p = stats.pearsonr(temp['NUMBER_OF_VEHICLES'], temp['AQI_PM10'])
    print(f"\nTest 2: Vehicle Count ↔ PM10")
    print(f"  Sample size: {len(temp):,}")
    print(f"  Pearson r: {r:.4f}")
    print(f"  P-value: {p:.6f}")
    print(f"  Result: {interpret_pvalue(p)}")
    h1_results['vehicle_pm10'] = {'r': r, 'p': p, 'significant': p < 0.05}

# Test 3: Traffic Density ↔ NO2
if 'traffic_density' in df.columns and 'Concentration_NO2' in df.columns:
    temp = df[['traffic_density', 'Concentration_NO2']].dropna()
    r, p = stats.pearsonr(temp['traffic_density'], temp['Concentration_NO2'])
    print(f"\nTest 3: Traffic Density ↔ NO2")
    print(f"  Sample size: {len(temp):,}")
    print(f"  Pearson r: {r:.4f}")
    print(f"  P-value: {p:.6f}")
    print(f"  Result: {interpret_pvalue(p)}")
    h1_results['density_no2'] = {'r': r, 'p': p, 'significant': p < 0.05}

# H1 Summary
sig_count = sum(1 for v in h1_results.values() if v['significant'])
print(f"\n{'='*80}")
print(f"H1 SUMMARY:")
print(f"  Significant tests: {sig_count}/{len(h1_results)}")
if sig_count > 0:
    print(f"  Result: SUPPORTED ({sig_count}/{len(h1_results)} tests significant)")
else:
    print(f"  Result: NOT SUPPORTED (no significant correlations)")
print(f"{'='*80}")

# =============================================================================
# H2: TIME LAG EFFECT (2-HOUR HYPOTHESIS)
# =============================================================================

print("\n" + "="*80)
print("H2: TIME LAG EFFECT ANALYSIS")
print("="*80)
print("Testing hypothesis: Traffic impacts pollution with 2-hour delay\n")

lag_results = []

# Test 0: Immediate (no lag)
temp = df[['NUMBER_OF_VEHICLES', 'Concentration_NO2']].dropna()
if len(temp) > 30:
    r, p = stats.pearsonr(temp['NUMBER_OF_VEHICLES'], temp['Concentration_NO2'])
    lag_results.append({'lag_hours': 0, 'r': r, 'p': p, 'significant': p < 0.05})
    sig = "significant" if p < 0.05 else "not significant"
    print(f"Lag 0h (immediate): r={r:.4f}, p={p:.6f} ({sig})")

# Test 1-3: Use enrichment lag features
for lag in [1, 2, 3]:
    lag_col = f'no2_lag{lag}'
    if lag_col in df.columns:
        temp = df[['NUMBER_OF_VEHICLES', lag_col]].dropna()
        if len(temp) > 30:
            r, p = stats.pearsonr(temp['NUMBER_OF_VEHICLES'], temp[lag_col])
            lag_results.append({'lag_hours': lag, 'r': r, 'p': p, 'significant': p < 0.05})
            sig = "significant" if p < 0.05 else "not significant"
            print(f"Lag {lag}h: r={r:.4f}, p={p:.6f} ({sig})")

# Test 4-6: Calculate additional lags
for lag in range(4, 7):
    df[f'no2_future_{lag}'] = df['Concentration_NO2'].shift(-lag)
    temp = df[['NUMBER_OF_VEHICLES', f'no2_future_{lag}']].dropna()
    
    if len(temp) > 30:
        r, p = stats.pearsonr(temp['NUMBER_OF_VEHICLES'], temp[f'no2_future_{lag}'])
        lag_results.append({'lag_hours': lag, 'r': r, 'p': p, 'significant': p < 0.05})
        sig = "significant" if p < 0.05 else "not significant"
        print(f"Lag {lag}h: r={r:.4f}, p={p:.6f} ({sig})")

# Find optimal lag
lag_df = pd.DataFrame(lag_results)
best_idx = lag_df['r'].abs().idxmax()
best_lag = lag_df.loc[best_idx]

print(f"\n{'='*80}")
print(f"H2 SUMMARY:")
print(f"  Optimal lag period: {best_lag['lag_hours']} hours")
print(f"  Correlation: r = {best_lag['r']:.4f}")
print(f"  P-value: p = {best_lag['p']:.6f}")
print(f"  Result: {interpret_pvalue(best_lag['p'])}")

if best_lag['lag_hours'] == 2 and best_lag['significant']:
    print(f"  Conclusion: SUPPORTED (2-hour lag optimal)")
elif best_lag['significant']:
    print(f"  Conclusion: PARTIALLY SUPPORTED ({best_lag['lag_hours']}h lag optimal, not 2h)")
else:
    print(f"  Conclusion: NOT SUPPORTED (no significant lag effect)")
print(f"{'='*80}")

# =============================================================================
# H3: HOLIDAY/SPECIAL DAY EFFECT
# =============================================================================

print("\n" + "="*80)
print("H3: HOLIDAY/SPECIAL DAY EFFECT")
print("="*80)

normal = df[df['is_special_day'] == 0]
special = df[df['is_special_day'] == 1]

print(f"\nSample Sizes:")
print(f"  Normal days: {len(normal):,}")
print(f"  Special days: {len(special):,}")

# Test 1: Vehicle Count
if 'NUMBER_OF_VEHICLES' in df.columns:
    normal_veh = normal['NUMBER_OF_VEHICLES'].dropna()
    special_veh = special['NUMBER_OF_VEHICLES'].dropna()
    
    t_stat, p_val = stats.ttest_ind(normal_veh, special_veh)
    d, d_interp = cohens_d(normal_veh, special_veh)
    
    print(f"\nTest 1: Vehicle Count Comparison")
    print(f"  Normal days mean: {normal_veh.mean():.1f} vehicles")
    print(f"  Special days mean: {special_veh.mean():.1f} vehicles")
    print(f"  Difference: {normal_veh.mean() - special_veh.mean():.1f} vehicles")
    print(f"  Percentage change: {((special_veh.mean() - normal_veh.mean()) / normal_veh.mean() * 100):.1f}%")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  P-value: {p_val:.6f}")
    print(f"  Cohen's d: {d:.4f} ({d_interp} effect)")
    print(f"  Result: {interpret_pvalue(p_val)}")
    h3_veh_sig = p_val < 0.05
else:
    h3_veh_sig = False

# Test 2: NO2 Concentration
if 'Concentration_NO2' in df.columns:
    normal_no2 = normal['Concentration_NO2'].dropna()
    special_no2 = special['Concentration_NO2'].dropna()
    
    t_stat, p_val = stats.ttest_ind(normal_no2, special_no2)
    d, d_interp = cohens_d(normal_no2, special_no2)
    
    print(f"\nTest 2: NO2 Concentration Comparison")
    print(f"  Normal days mean: {normal_no2.mean():.2f} µg/m³")
    print(f"  Special days mean: {special_no2.mean():.2f} µg/m³")
    print(f"  Difference: {normal_no2.mean() - special_no2.mean():.2f} µg/m³")
    print(f"  Percentage change: {((special_no2.mean() - normal_no2.mean()) / normal_no2.mean() * 100):.1f}%")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  P-value: {p_val:.6f}")
    print(f"  Cohen's d: {d:.4f} ({d_interp} effect)")
    print(f"  Result: {interpret_pvalue(p_val)}")
    h3_no2_sig = p_val < 0.05
else:
    h3_no2_sig = False

print(f"\n{'='*80}")
print(f"H3 SUMMARY:")
if h3_veh_sig or h3_no2_sig:
    print(f"  Result: SUPPORTED ({sum([h3_veh_sig, h3_no2_sig])}/2 tests significant)")
else:
    print(f"  Result: NOT SUPPORTED (no significant differences)")
print(f"{'='*80}")

# =============================================================================
# H4: WEEKEND EFFECT
# =============================================================================

print("\n" + "="*80)
print("H4: WEEKEND EFFECT")
print("="*80)

weekday = df[df['is_weekend'] == 0]
weekend = df[df['is_weekend'] == 1]

print(f"\nSample Sizes:")
print(f"  Weekdays: {len(weekday):,}")
print(f"  Weekends: {len(weekend):,}")

# Test 1: Vehicle Count
if 'NUMBER_OF_VEHICLES' in df.columns:
    weekday_veh = weekday['NUMBER_OF_VEHICLES'].dropna()
    weekend_veh = weekend['NUMBER_OF_VEHICLES'].dropna()
    
    t_stat, p_val = stats.ttest_ind(weekday_veh, weekend_veh)
    d, d_interp = cohens_d(weekday_veh, weekend_veh)
    
    print(f"\nTest 1: Vehicle Count Comparison")
    print(f"  Weekday mean: {weekday_veh.mean():.1f} vehicles")
    print(f"  Weekend mean: {weekend_veh.mean():.1f} vehicles")
    print(f"  Difference: {weekday_veh.mean() - weekend_veh.mean():.1f} vehicles")
    print(f"  Percentage change: {((weekend_veh.mean() - weekday_veh.mean()) / weekday_veh.mean() * 100):.1f}%")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  P-value: {p_val:.6f}")
    print(f"  Cohen's d: {d:.4f} ({d_interp} effect)")
    print(f"  Result: {interpret_pvalue(p_val)}")
    h4_veh_sig = p_val < 0.05
else:
    h4_veh_sig = False

# Test 2: NO2 Concentration
if 'Concentration_NO2' in df.columns:
    weekday_no2 = weekday['Concentration_NO2'].dropna()
    weekend_no2 = weekend['Concentration_NO2'].dropna()
    
    t_stat, p_val = stats.ttest_ind(weekday_no2, weekend_no2)
    d, d_interp = cohens_d(weekday_no2, weekend_no2)
    
    print(f"\nTest 2: NO2 Concentration Comparison")
    print(f"  Weekday mean: {weekday_no2.mean():.2f} µg/m³")
    print(f"  Weekend mean: {weekend_no2.mean():.2f} µg/m³")
    print(f"  Difference: {weekday_no2.mean() - weekend_no2.mean():.2f} µg/m³")
    print(f"  Percentage change: {((weekend_no2.mean() - weekday_no2.mean()) / weekday_no2.mean() * 100):.1f}%")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  P-value: {p_val:.6f}")
    print(f"  Cohen's d: {d:.4f} ({d_interp} effect)")
    print(f"  Result: {interpret_pvalue(p_val)}")
    h4_no2_sig = p_val < 0.05
else:
    h4_no2_sig = False

print(f"\n{'='*80}")
print(f"H4 SUMMARY:")
if h4_veh_sig or h4_no2_sig:
    print(f"  Result: SUPPORTED ({sum([h4_veh_sig, h4_no2_sig])}/2 tests significant)")
else:
    print(f"  Result: NOT SUPPORTED (no significant differences)")
print(f"{'='*80}")

# =============================================================================
# H5: RUSH HOUR POLLUTION PEAKS
# =============================================================================

print("\n" + "="*80)
print("H5: RUSH HOUR POLLUTION PEAKS")
print("="*80)

morning = df[df['rush_hour'] == 'Sabah Rush (7-9)']
evening = df[df['rush_hour'] == 'Akşam Rush (17-19)']
other = df[df['rush_hour'] == 'Diğer Saatler']

print(f"\nSample Sizes:")
print(f"  Morning rush (7-9h): {len(morning):,}")
print(f"  Evening rush (17-19h): {len(evening):,}")
print(f"  Other hours: {len(other):,}")

if len(morning) > 0 and len(evening) > 0 and len(other) > 0:
    morning_no2 = morning['Concentration_NO2'].dropna()
    evening_no2 = evening['Concentration_NO2'].dropna()
    other_no2 = other['Concentration_NO2'].dropna()
    
    if len(morning_no2) > 0 and len(evening_no2) > 0 and len(other_no2) > 0:
        print(f"\nNO2 Concentration by Time Period:")
        print(f"  Morning rush: {morning_no2.mean():.2f} µg/m³ (SD: {morning_no2.std():.2f})")
        print(f"  Evening rush: {evening_no2.mean():.2f} µg/m³ (SD: {evening_no2.std():.2f})")
        print(f"  Other hours: {other_no2.mean():.2f} µg/m³ (SD: {other_no2.std():.2f})")
        
        # One-way ANOVA
        f_stat, p_val = stats.f_oneway(morning_no2, evening_no2, other_no2)
        
        print(f"\nOne-way ANOVA Results:")
        print(f"  F-statistic: {f_stat:.4f}")
        print(f"  P-value: {p_val:.6f}")
        print(f"  Result: {interpret_pvalue(p_val)}")
        
        # Post-hoc: pairwise comparisons
        if p_val < 0.05:
            print(f"\nPost-hoc Pairwise Comparisons:")
            t_m_e, p_m_e = stats.ttest_ind(morning_no2, evening_no2)
            print(f"  Morning vs Evening: t={t_m_e:.4f}, p={p_m_e:.6f} ({interpret_pvalue(p_m_e)})")
            
            t_m_o, p_m_o = stats.ttest_ind(morning_no2, other_no2)
            print(f"  Morning vs Other: t={t_m_o:.4f}, p={p_m_o:.6f} ({interpret_pvalue(p_m_o)})")
            
            t_e_o, p_e_o = stats.ttest_ind(evening_no2, other_no2)
            print(f"  Evening vs Other: t={t_e_o:.4f}, p={p_e_o:.6f} ({interpret_pvalue(p_e_o)})")
        
        print(f"\n{'='*80}")
        if p_val < 0.05:
            print(f"H5 SUMMARY:")
            print(f"  Result: SUPPORTED (significant differences across time periods)")
            h5_supported = True
        else:
            print(f"H5 SUMMARY:")
            print(f"  Result: NOT SUPPORTED (no significant rush hour effect)")
            h5_supported = False
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print(f"H5 SUMMARY:")
        print(f"  Result: INCONCLUSIVE (insufficient data in some categories)")
        print(f"{'='*80}")
        h5_supported = None
else:
    print(f"\n{'='*80}")
    print(f"H5 SUMMARY:")
    print(f"  Result: INCONCLUSIVE (missing rush hour data)")
    print(f"{'='*80}")
    h5_supported = None

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*80)
print("HYPOTHESIS TESTING - FINAL SUMMARY")
print("="*80)

print("\nAll Hypothesis Results:")
print(f"\n  H1 (Traffic ↔ Pollution): ", end="")
if sig_count > 0:
    print(f"SUPPORTED ({sig_count}/{len(h1_results)} tests significant)")
else:
    print(f"NOT SUPPORTED (0/{len(h1_results)} tests significant)")

print(f"  H2 (2-hour lag effect): ", end="")
if best_lag['lag_hours'] == 2 and best_lag['significant']:
    print("SUPPORTED (2h lag optimal)")
elif best_lag['significant']:
    print(f"PARTIALLY SUPPORTED ({best_lag['lag_hours']}h lag optimal, not 2h)")
else:
    print("NOT SUPPORTED (no significant lag)")

print(f"  H3 (Holiday effect): ", end="")
if h3_veh_sig or h3_no2_sig:
    print(f"SUPPORTED ({sum([h3_veh_sig, h3_no2_sig])}/2 tests significant)")
else:
    print("NOT SUPPORTED (no significant differences)")

print(f"  H4 (Weekend effect): ", end="")
if h4_veh_sig or h4_no2_sig:
    print(f"SUPPORTED ({sum([h4_veh_sig, h4_no2_sig])}/2 tests significant)")
else:
    print("NOT SUPPORTED (no significant differences)")

print(f"  H5 (Rush hour peaks): ", end="")
if h5_supported == True:
    print("SUPPORTED (significant ANOVA)")
elif h5_supported == False:
    print("NOT SUPPORTED (no significant differences)")
else:
    print("INCONCLUSIVE (insufficient data)")

print("\n" + "="*80)
print("HYPOTHESIS TESTING COMPLETED SUCCESSFULLY")
print("="*80)
