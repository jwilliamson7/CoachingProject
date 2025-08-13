import pandas as pd
import numpy as np
from collections import defaultdict

def detailed_data_comparison():
    """Comprehensive comparison between old and new master_data files"""
    
    print("="*80)
    print("DETAILED MASTER DATA COMPARISON")
    print("="*80)
    
    # Load both datasets
    try:
        old_df = pd.read_csv('old_master_data.csv')
        new_df = pd.read_csv('master_data.csv')
        print(f"* Loaded old_master_data.csv: {len(old_df)} rows, {len(old_df.columns)} columns")
        print(f"* Loaded master_data.csv: {len(new_df)} rows, {len(new_df.columns)} columns")
    except FileNotFoundError as e:
        print(f"ERROR loading files: {e}")
        return
    
    print("\n" + "="*60)
    print("1. BASIC DATASET COMPARISON")
    print("="*60)
    
    # Basic shape comparison
    print(f"Rows: {len(old_df)} -> {len(new_df)} (change: {len(new_df) - len(old_df):+d})")
    print(f"Columns: {len(old_df.columns)} -> {len(new_df.columns)} (change: {len(new_df.columns) - len(old_df.columns):+d})")
    
    # Column comparison
    old_cols = set(old_df.columns)
    new_cols = set(new_df.columns)
    
    if old_cols != new_cols:
        print(f"\n📋 Column differences:")
        added_cols = new_cols - old_cols
        removed_cols = old_cols - new_cols
        if added_cols:
            print(f"  Added columns: {list(added_cols)}")
        if removed_cols:
            print(f"  Removed columns: {list(removed_cols)}")
    else:
        print("* Column names identical")
    
    print("\n" + "="*60)
    print("2. COACH-LEVEL COMPARISON")
    print("="*60)
    
    # Compare coaches present
    old_coaches = set(old_df['Coach Name'].unique())
    new_coaches = set(new_df['Coach Name'].unique())
    
    print(f"Coaches: {len(old_coaches)} -> {len(new_coaches)} (change: {len(new_coaches) - len(old_coaches):+d})")
    
    added_coaches = new_coaches - old_coaches
    removed_coaches = old_coaches - new_coaches
    
    if added_coaches:
        print(f"  Added coaches: {list(added_coaches)[:10]}{'...' if len(added_coaches) > 10 else ''}")
    if removed_coaches:
        print(f"  Removed coaches: {list(removed_coaches)[:10]}{'...' if len(removed_coaches) > 10 else ''}")
    
    # Check for coaches with different instance counts
    old_counts = old_df['Coach Name'].value_counts().to_dict()
    new_counts = new_df['Coach Name'].value_counts().to_dict()
    
    changed_instances = []
    for coach in old_coaches & new_coaches:
        old_count = old_counts[coach]
        new_count = new_counts.get(coach, 0)
        if old_count != new_count:
            changed_instances.append((coach, old_count, new_count))
    
    if changed_instances:
        print(f"\nCoaches with different instance counts:")
        for coach, old_count, new_count in changed_instances[:10]:
            print(f"  {coach}: {old_count} -> {new_count}")
        if len(changed_instances) > 10:
            print(f"  ... and {len(changed_instances) - 10} more")
    else:
        print("* All coaches have same number of instances")
    
    print("\n" + "="*60)
    print("3. BILL BELICHICK DETAILED COMPARISON")
    print("="*60)
    
    # Focus on Belichick since that's what we were debugging
    old_bb = old_df[old_df['Coach Name'] == 'Bill Belichick'].copy()
    new_bb = new_df[new_df['Coach Name'] == 'Bill Belichick'].copy()
    
    print(f"Bill Belichick instances: {len(old_bb)} -> {len(new_bb)}")
    
    if len(old_bb) > 0 and len(new_bb) > 0:
        # Compare each instance
        for i in range(min(len(old_bb), len(new_bb))):
            old_instance = old_bb.iloc[i]
            new_instance = new_bb.iloc[i]
            
            print(f"\n  Instance {i+1} (Year {old_instance['Year']} vs {new_instance['Year']}):")
            
            # Check core features
            core_features = ['Year', 'Feature 1', 'Feature 2', 'Feature 7', 'Feature 8']
            for feature in core_features:
                if feature in old_instance.index and feature in new_instance.index:
                    old_val = old_instance[feature]
                    new_val = new_instance[feature]
                    if pd.isna(old_val) and pd.isna(new_val):
                        status = "OK"
                    elif pd.isna(old_val) != pd.isna(new_val):
                        status = "DIFF"
                    elif isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                        if abs(old_val - new_val) < 1e-10:
                            status = "OK"
                        else:
                            status = "DIFF"
                    elif old_val == new_val:
                        status = "OK"
                    else:
                        status = "DIFF"
                    print(f"    {feature}: {old_val} -> {new_val} {status}")
            
            # Check DC features (62-74) - the ones we were investigating
            dc_features = [f'Feature {i}' for i in range(62, 75)]
            old_dc_data = sum(1 for f in dc_features if f in old_instance.index and not pd.isna(old_instance[f]))
            new_dc_data = sum(1 for f in dc_features if f in new_instance.index and not pd.isna(new_instance[f]))
            print(f"    DC features (62-74) with data: {old_dc_data} -> {new_dc_data}")
            
            # Check HC features (95-140) 
            hc_features = [f'Feature {i}' for i in range(95, 141)]
            old_hc_data = sum(1 for f in hc_features if f in old_instance.index and not pd.isna(old_instance[f]))
            new_hc_data = sum(1 for f in hc_features if f in new_instance.index and not pd.isna(new_instance[f]))
            print(f"    HC features (95-140) with data: {old_hc_data} -> {new_hc_data}")
    
    print("\n" + "="*60)
    print("4. FEATURE-LEVEL STATISTICAL COMPARISON")
    print("="*60)
    
    # Compare key numerical features statistically
    key_features = ['Feature 1', 'Feature 2', 'Feature 7', 'Feature 8', 'Avg 2Y Win Pct']
    
    for feature in key_features:
        if feature in old_df.columns and feature in new_df.columns:
            old_vals = old_df[feature].dropna()
            new_vals = new_df[feature].dropna()
            
            print(f"\n{feature}:")
            print(f"  Count: {len(old_vals)} -> {len(new_vals)}")
            print(f"  Mean: {old_vals.mean():.4f} -> {new_vals.mean():.4f}")
            print(f"  Std: {old_vals.std():.4f} -> {new_vals.std():.4f}")
            print(f"  Min: {old_vals.min():.4f} -> {new_vals.min():.4f}")
            print(f"  Max: {old_vals.max():.4f} -> {new_vals.max():.4f}")
    
    print("\n" + "="*60)
    print("5. MISSING DATA PATTERN COMPARISON")
    print("="*60)
    
    # Compare NaN patterns for the problematic feature ranges
    feature_ranges = [
        ("DC Features 41-53", range(41, 54)),
        ("DC Features 62-74", range(62, 75)), 
        ("HC Features 74-94", range(74, 95)),
        ("HC Features 95-140", range(95, 141))
    ]
    
    for range_name, feature_range in feature_ranges:
        feature_cols = [f'Feature {i}' for i in feature_range if f'Feature {i}' in old_df.columns]
        
        if feature_cols:
            old_missing = old_df[feature_cols].isna().sum().sum()
            new_missing = new_df[feature_cols].isna().sum().sum()
            old_total = len(old_df) * len(feature_cols)
            new_total = len(new_df) * len(feature_cols)
            
            old_pct = (old_missing / old_total * 100) if old_total > 0 else 0
            new_pct = (new_missing / new_total * 100) if new_total > 0 else 0
            
            print(f"{range_name}:")
            print(f"  Missing values: {old_missing}/{old_total} ({old_pct:.1f}%) -> {new_missing}/{new_total} ({new_pct:.1f}%)")
    
    print("\n" + "="*60)
    print("6. SPECIFIC COACH EXAMPLES")
    print("="*60)
    
    # Check a few other coaches who might have multiple tenures
    multi_tenure_coaches = ['Andy Reid', 'Sean Payton', 'Jon Gruden', 'Wade Phillips']
    
    for coach in multi_tenure_coaches:
        old_coach = old_df[old_df['Coach Name'] == coach]
        new_coach = new_df[new_df['Coach Name'] == coach]
        
        if len(old_coach) > 0 or len(new_coach) > 0:
            print(f"\n{coach}: {len(old_coach)} -> {len(new_coach)} instances")
            
            if len(old_coach) > 0 and len(new_coach) > 0:
                # Check if years match
                old_years = sorted(old_coach['Year'].tolist())
                new_years = sorted(new_coach['Year'].tolist())
                print(f"  Years: {old_years} -> {new_years}")
    
    print("\n" + "="*60)
    print("7. SUMMARY")
    print("="*60)
    
    # Overall assessment
    total_differences = 0
    
    # Count major differences
    if len(old_df) != len(new_df):
        total_differences += abs(len(old_df) - len(new_df))
        print(f"WARNING: Row count changed by {len(new_df) - len(old_df)}")
    
    if len(changed_instances) > 0:
        total_differences += len(changed_instances)
        print(f"WARNING: {len(changed_instances)} coaches have different instance counts")
    
    if added_coaches or removed_coaches:
        total_differences += len(added_coaches) + len(removed_coaches)
        print(f"WARNING: Coach set changed: +{len(added_coaches)} -{len(removed_coaches)}")
    
    if total_differences == 0:
        print("* No major structural differences detected")
        print("* Changes appear to be data quality improvements only")
    else:
        print(f"WARNING: {total_differences} differences detected - review recommended")
    
    print("\n" + "="*60)
    print("8. COMPREHENSIVE FEATURE-BY-FEATURE COMPARISON")
    print("="*60)
    
    # Get all feature columns (excluding experience features)
    experience_features = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5', 'Feature 6', 'Feature 7', 'Feature 8']
    feature_cols = [col for col in old_df.columns if col.startswith('Feature') and col not in experience_features]
    
    print(f"Comparing {len(feature_cols)} non-experience features across all coach-year instances...")
    
    # Create merged dataset on Coach Name + Year for comparison
    old_df_indexed = old_df.set_index(['Coach Name', 'Year'])
    new_df_indexed = new_df.set_index(['Coach Name', 'Year'])
    
    # Find matching coach-year combinations
    common_indices = old_df_indexed.index.intersection(new_df_indexed.index)
    print(f"Found {len(common_indices)} matching coach-year instances")
    
    coaches_with_diffs = []
    total_feature_diffs = 0
    total_comparisons = 0
    
    for coach_year in common_indices:
        old_row = old_df_indexed.loc[coach_year]
        new_row = new_df_indexed.loc[coach_year]
        
        coach_diffs = []
        
        for feature in feature_cols:
            if feature in old_row.index and feature in new_row.index:
                old_val = old_row[feature]
                new_val = new_row[feature]
                
                total_comparisons += 1
                
                # Check for significant differences
                diff_found = False
                
                if pd.isna(old_val) and pd.isna(new_val):
                    # Both NaN - no difference
                    continue
                elif pd.isna(old_val) != pd.isna(new_val):
                    # One NaN, one not - significant difference
                    diff_found = True
                    diff_type = "NaN_change"
                    diff_value = "NaN -> value" if pd.isna(old_val) else "value -> NaN"
                elif isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                    # Both numeric - check absolute difference
                    abs_diff = abs(old_val - new_val)
                    if abs_diff > 0.01:
                        diff_found = True
                        diff_type = "numeric"
                        diff_value = abs_diff
                elif old_val != new_val:
                    # Other type of difference
                    diff_found = True
                    diff_type = "other"
                    diff_value = f"{old_val} -> {new_val}"
                
                if diff_found:
                    total_feature_diffs += 1
                    coach_diffs.append({
                        'feature': feature,
                        'old_val': old_val,
                        'new_val': new_val,
                        'diff_type': diff_type,
                        'diff_value': diff_value
                    })
        
        if coach_diffs:
            coaches_with_diffs.append({
                'coach': coach_year[0],
                'year': coach_year[1],
                'num_diffs': len(coach_diffs),
                'diffs': coach_diffs
            })
    
    print(f"\nResults:")
    print(f"  Total feature comparisons: {total_comparisons:,}")
    print(f"  Features with differences > 0.01: {total_feature_diffs:,}")
    print(f"  Coaches with any feature differences: {len(coaches_with_diffs)}")
    print(f"  Difference rate: {total_feature_diffs/total_comparisons*100:.2f}%" if total_comparisons > 0 else "  Difference rate: N/A")
    
    if coaches_with_diffs:
        print(f"\nTop 20 coaches with most feature differences:")
        coaches_with_diffs.sort(key=lambda x: x['num_diffs'], reverse=True)
        
        for i, coach_info in enumerate(coaches_with_diffs[:20]):
            print(f"\n{i+1}. {coach_info['coach']} ({coach_info['year']}): {coach_info['num_diffs']} differences")
            
            # Show first 5 differences for this coach
            for j, diff in enumerate(coach_info['diffs'][:5]):
                if diff['diff_type'] == 'numeric':
                    print(f"   {diff['feature']}: {diff['old_val']:.6f} -> {diff['new_val']:.6f} (diff: {diff['diff_value']:.6f})")
                elif diff['diff_type'] == 'NaN_change':
                    print(f"   {diff['feature']}: {diff['diff_value']}")
                else:
                    print(f"   {diff['feature']}: {diff['diff_value']}")
            
            if len(coach_info['diffs']) > 5:
                print(f"   ... and {len(coach_info['diffs']) - 5} more differences")
    
    # Analyze patterns in differences
    if coaches_with_diffs:
        print(f"\n" + "="*50)
        print("DIFFERENCE PATTERN ANALYSIS")
        print("="*50)
        
        # Count differences by feature
        feature_diff_counts = defaultdict(int)
        nan_change_counts = defaultdict(int)
        
        for coach_info in coaches_with_diffs:
            for diff in coach_info['diffs']:
                feature_diff_counts[diff['feature']] += 1
                if diff['diff_type'] == 'NaN_change':
                    nan_change_counts[diff['feature']] += 1
        
        print(f"Features with most differences:")
        sorted_features = sorted(feature_diff_counts.items(), key=lambda x: x[1], reverse=True)
        for feature, count in sorted_features[:15]:
            nan_count = nan_change_counts.get(feature, 0)
            print(f"  {feature}: {count} differences ({nan_count} NaN changes)")
        
        # Check if differences cluster around specific feature ranges
        dc_diffs = sum(count for feature, count in feature_diff_counts.items() if "__dc" in feature)
        oc_diffs = sum(count for feature, count in feature_diff_counts.items() if "__oc" in feature)
        hc_diffs = sum(count for feature, count in feature_diff_counts.items() if "__hc" in feature)
        
        print(f"\nDifferences by role:")
        print(f"  Defensive Coordinator features: {dc_diffs}")
        print(f"  Offensive Coordinator features: {oc_diffs}")
        print(f"  Head Coach features: {hc_diffs}")
    
    print("\nKey takeaways:")
    print("   - Focus on whether Bill Belichick's 1991 instance now has better DC feature coverage")
    print("   - Verify that overall statistics and distributions remain stable")
    print("   - Missing data patterns should reflect historical NFL statistics evolution")
    print("   - Large numbers of feature differences may indicate data processing changes")

if __name__ == "__main__":
    detailed_data_comparison()