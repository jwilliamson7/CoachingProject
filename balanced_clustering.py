import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

def force_balanced_clustering(data, n_clusters=5, balance_penalty=2.0):
    """Force more balanced clustering using all available features"""
    
    # Use all features (pathway + performance)
    all_features = [f'Feature {i}' for i in range(1, 151)]
    
    # Filter to modern era (1970+) for better data quality
    modern_data = data[data['Year'] >= 1970].copy()
    print(f"Filtered to modern era (1970+): {len(modern_data)} coaches ({len(modern_data)/len(data)*100:.1f}% of total)")
    
    # Check feature completeness in modern era
    good_features = []
    feature_completeness = {}
    
    for feature in all_features:
        if feature in modern_data.columns:
            completeness = (modern_data[feature].notna().sum() / len(modern_data)) * 100
            feature_completeness[feature] = completeness
            if completeness > 30:  # Lower threshold for modern era
                good_features.append(feature)
    
    pathway_features = [f for f in good_features if f in [f'Feature {i}' for i in range(1, 9)]]
    performance_features = [f for f in good_features if f in [f'Feature {i}' for i in range(9, 151)]]
    
    print(f"Features with >30% completeness:")
    print(f"  Pathway features: {len(pathway_features)}/8")
    print(f"  Performance features: {len(performance_features)}/142")
    print(f"  Total usable features: {len(good_features)}")
    
    # Clean and prepare data
    clean_data = modern_data[good_features].copy()
    for col in good_features:
        clean_data[col] = clean_data[col].fillna(clean_data[col].median())
    
    # Scale data
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(clean_data)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=0.95, random_state=42)  # Keep 95% of variance
    pca_data = pca.fit_transform(scaled_data)
    
    print(f"PCA: {len(good_features)} features -> {pca_data.shape[1]} components ({pca.explained_variance_ratio_.sum():.1%} variance)")
    print(f"Forcing {n_clusters} balanced clusters on all available features...")
    
    best_labels = None
    best_score = -1
    best_balance = 0
    
    # Try multiple random initializations to find balanced solution
    for trial in range(50):
        kmeans = KMeans(n_clusters=n_clusters, random_state=trial, n_init=20, init='k-means++')
        labels = kmeans.fit_predict(pca_data)
        
        # Calculate cluster balance
        cluster_sizes = np.bincount(labels)
        balance_score = 1 - (np.std(cluster_sizes) / np.mean(cluster_sizes))
        balance_score = max(0, balance_score)
        
        # Calculate silhouette score
        sil_score = silhouette_score(pca_data, labels)
        
        # Combined score that heavily weights balance
        combined_score = sil_score + (balance_penalty * balance_score)
        
        if combined_score > best_score:
            best_score = combined_score
            best_labels = labels.copy()
            best_balance = balance_score
            best_sil = sil_score
    
    print(f"Best solution - Silhouette: {best_sil:.3f}, Balance: {best_balance:.3f}, Combined: {best_score:.3f}")
    
    return best_labels, pca_data, good_features, modern_data

def analyze_balanced_clusters(data, labels, features, feature_names):
    """Analyze the balanced clustering results"""
    
    print(f"\n{'='*60}")
    print("BALANCED CLUSTERING RESULTS")
    print(f"{'='*60}")
    
    unique_labels = np.unique(labels)
    cluster_sizes = np.bincount(labels)
    
    print(f"\nCluster size distribution:")
    for i, (label, size) in enumerate(zip(unique_labels, cluster_sizes)):
        pct = size / len(labels) * 100
        print(f"Cluster {label}: {size} coaches ({pct:.1f}%)")
    
    # Balance metrics
    mean_size = np.mean(cluster_sizes)
    std_size = np.std(cluster_sizes)
    cv = std_size / mean_size
    max_min_ratio = max(cluster_sizes) / min(cluster_sizes)
    
    print(f"\nBalance Metrics:")
    print(f"Mean cluster size: {mean_size:.1f}")
    print(f"Standard deviation: {std_size:.1f}")
    print(f"Coefficient of variation: {cv:.3f}")
    print(f"Max/min size ratio: {max_min_ratio:.1f}")
    
    # Detailed cluster analysis
    for cluster in unique_labels:
        mask = labels == cluster
        cluster_data = data[mask]
        
        print(f"\n{'='*40}")
        print(f"CLUSTER {cluster} PROFILE ({len(cluster_data)} coaches)")
        print(f"{'='*40}")
        
        # Core characteristics
        print("Defining Characteristics:")
        print(f"  Average Age: {cluster_data['Feature 1'].mean():.1f} years")
        print(f"  Previous HC Stints: {cluster_data['Feature 2'].mean():.1f}")
        print(f"  College Position Coach: {cluster_data['Feature 3'].mean():.1f} years")
        print(f"  College Coordinator: {cluster_data['Feature 4'].mean():.1f} years")
        print(f"  College Head Coach: {cluster_data['Feature 5'].mean():.1f} years")
        print(f"  NFL Position Coach: {cluster_data['Feature 6'].mean():.1f} years")
        print(f"  NFL Coordinator: {cluster_data['Feature 7'].mean():.1f} years")
        print(f"  NFL Head Coach: {cluster_data['Feature 8'].mean():.1f} years")
        
        # Success metrics (post-clustering reference)
        known_outcomes = cluster_data[cluster_data['Coach Tenure Class'] != -1]
        if len(known_outcomes) > 0:
            avg_win_pct = known_outcomes['Avg 2Y Win Pct'].mean()
            print(f"\nOutcome Metrics (reference only):")
            print(f"  Average Win %: {avg_win_pct:.3f}")
            
            tenure_dist = known_outcomes['Coach Tenure Class'].value_counts().sort_index()
            for tenure, count in tenure_dist.items():
                pct = count / len(known_outcomes) * 100
                tenure_label = ['<=2yr', '3-4yr', '5+yr'][tenure]
                print(f"    {tenure_label}: {count} ({pct:.1f}%)")
        
        # Recent hires
        recent = cluster_data[cluster_data['Coach Tenure Class'] == -1]
        if len(recent) > 0:
            print(f"  Recent hires: {len(recent)}")
            names = recent['Coach Name'].head(3).tolist()
            print(f"    Examples: {', '.join(names)}")
        
        # Cluster interpretation
        age_mean = cluster_data['Feature 1'].mean()
        college_coord = cluster_data['Feature 4'].mean()
        nfl_position = cluster_data['Feature 6'].mean()
        nfl_coord = cluster_data['Feature 7'].mean()
        
        # Generate cluster description
        if age_mean < 40:
            age_desc = "Very young"
        elif age_mean < 45:
            age_desc = "Young"
        elif age_mean < 50:
            age_desc = "Middle-aged"
        else:
            age_desc = "Veteran"
            
        # Determine dominant background
        college_total = cluster_data['Feature 3'].mean() + cluster_data['Feature 4'].mean() + cluster_data['Feature 5'].mean()
        nfl_total = cluster_data['Feature 6'].mean() + cluster_data['Feature 7'].mean() + cluster_data['Feature 8'].mean()
        
        if college_total > nfl_total:
            background_desc = "college coaching background"
        else:
            background_desc = "NFL coaching background"
            
        if nfl_coord > 3:
            coord_desc = "strong NFL coordinator experience"
        elif college_coord > 3:
            coord_desc = "college coordinator experience"
        elif nfl_position > 5:
            coord_desc = "NFL position coach experience"
        else:
            coord_desc = "mixed experience levels"
            
        print(f"\nCluster Profile: {age_desc} coaches with {background_desc} and {coord_desc}")

def create_balanced_visualization(data, labels, pca_features, feature_names):
    """Create visualizations for balanced clustering"""
    
    # Create PCA for visualization (2D from existing PCA features)
    pca_2d = PCA(n_components=2, random_state=42)
    pca_vis = pca_2d.fit_transform(pca_features)
    
    # Create t-SNE for visualization
    print("Computing t-SNE visualization...")
    perplexity = min(30, len(pca_features) // 8)  # Adjust perplexity based on data size
    
    # Compute both 2D and 3D t-SNE
    tsne_2d = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                   n_iter=500, learning_rate=200)
    tsne_features = tsne_2d.fit_transform(pca_features)
    
    print("Computing 3D t-SNE for interactive visualization...")
    tsne_3d = TSNE(n_components=3, random_state=42, perplexity=perplexity, 
                   n_iter=500, learning_rate=200)
    tsne_3d_features = tsne_3d.fit_transform(pca_features)
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    unique_labels = np.unique(labels)
    cluster_sizes = [np.sum(labels == label) for label in unique_labels]
    total_coaches = len(labels)
    
    # 1. PCA cluster plot
    scatter = axes[0,0].scatter(pca_vis[:, 0], pca_vis[:, 1], 
                              c=labels, cmap='Set3', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[0,0].set_title('Balanced Coach Clusters\n(PCA Visualization)', fontsize=11, fontweight='bold')
    axes[0,0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
    axes[0,0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
    plt.colorbar(scatter, ax=axes[0,0], label='Cluster')
    
    # 2. t-SNE cluster plot
    scatter2 = axes[0,1].scatter(tsne_features[:, 0], tsne_features[:, 1], 
                               c=labels, cmap='Set3', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[0,1].set_title('Balanced Coach Clusters\n(t-SNE Visualization)', fontsize=11, fontweight='bold')
    axes[0,1].set_xlabel('t-SNE 1')
    axes[0,1].set_ylabel('t-SNE 2')
    plt.colorbar(scatter2, ax=axes[0,1], label='Cluster')
    
    # 3. Cluster size distribution
    bars = axes[0,2].bar([f'C{i}' for i in unique_labels], cluster_sizes, 
                        color='skyblue', alpha=0.7, edgecolor='black')
    axes[0,2].set_title('Cluster Size Distribution', fontsize=11, fontweight='bold')
    axes[0,2].set_ylabel('Number of Coaches')
    axes[0,2].set_xlabel('Cluster')
    
    # Add percentage labels
    for bar, size in zip(bars, cluster_sizes):
        height = bar.get_height()
        pct = (size / total_coaches) * 100
        axes[0,2].text(bar.get_x() + bar.get_width()/2., height + 2,
                      f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. Age vs Experience colored by cluster (PCA view)
    axes[1,0].scatter(data['Feature 1'], data['Feature 7'], c=labels, cmap='Set3', 
                     s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[1,0].set_title('Age vs NFL Coordinator Experience\n(by Cluster)', fontsize=11, fontweight='bold')
    axes[1,0].set_xlabel('Age at Hire')
    axes[1,0].set_ylabel('NFL Coordinator Experience (Years)')
    
    # 5. Experience profile by cluster
    experience_features = ['Feature 3', 'Feature 4', 'Feature 5', 'Feature 6', 'Feature 7', 'Feature 8']
    exp_labels = ['Col Position', 'Col Coord', 'Col HC', 'NFL Position', 'NFL Coord', 'NFL HC']
    
    for cluster in unique_labels:
        mask = labels == cluster
        cluster_exp = data[mask][experience_features].mean()
        axes[1,1].plot(exp_labels, cluster_exp.values, 'o-', linewidth=2, 
                      markersize=8, label=f'Cluster {cluster}', alpha=0.8)
    
    axes[1,1].set_title('Experience Profiles by Cluster', fontsize=11, fontweight='bold')
    axes[1,1].set_xlabel('Experience Type')
    axes[1,1].set_ylabel('Average Years')
    axes[1,1].legend()
    axes[1,1].tick_params(axis='x', rotation=45)
    
    # Adjust bottom margin for rotated labels
    plt.setp(axes[1,1].xaxis.get_majorticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # 6. Win percentage distribution by cluster (for known outcomes)
    known_outcomes = data['Coach Tenure Class'] != -1
    if known_outcomes.any():
        for cluster in unique_labels:
            mask = (labels == cluster) & known_outcomes
            if mask.any():
                win_pcts = data[mask]['Avg 2Y Win Pct']
                axes[1,2].hist(win_pcts, bins=10, alpha=0.6, label=f'Cluster {cluster}')
        
        axes[1,2].set_title('Win % Distribution by Cluster\n(Known Outcomes)', fontsize=11, fontweight='bold')
        axes[1,2].set_xlabel('2-Year Win Percentage')
        axes[1,2].set_ylabel('Count')
        axes[1,2].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='0.500')
        axes[1,2].legend()
    
    # 7. Cluster characteristics heatmap (using core interpretable features)
    core_features = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 
                    'Feature 5', 'Feature 6', 'Feature 7', 'Feature 8']
    available_core = [f for f in core_features if f in data.columns]
    
    cluster_means = []
    for cluster in unique_labels:
        mask = labels == cluster
        cluster_mean = data[mask][available_core].mean()
        cluster_means.append(cluster_mean)
    
    cluster_df = pd.DataFrame(cluster_means, index=[f'Cluster {i}' for i in unique_labels])
    cluster_df_norm = (cluster_df - cluster_df.mean()) / cluster_df.std()  # Standardize for heatmap
    
    sns.heatmap(cluster_df_norm.T, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=axes[2,0], cbar_kws={'label': 'Standardized Value'})
    axes[2,0].set_title('Core Cluster Characteristics\n(Standardized)', fontsize=11, fontweight='bold')
    axes[2,0].set_xlabel('Cluster')
    axes[2,0].set_ylabel('Feature')
    
    # Set y-axis labels for better readability
    feature_labels = ['Age', 'Prev HC', 'Col Position', 'Col Coord', 'Col HC', 'NFL Position', 'NFL Coord', 'NFL HC']
    axes[2,0].set_yticklabels(feature_labels[:len(available_core)], rotation=0)
    
    # 8. Age distribution by cluster
    age_data = [data[labels == i]['Feature 1'].values for i in unique_labels]
    bp = axes[2,1].boxplot(age_data, labels=[f'C{i}' for i in unique_labels], patch_artist=True)
    axes[2,1].set_title('Age Distribution by Cluster', fontsize=11, fontweight='bold')
    axes[2,1].set_xlabel('Cluster')
    axes[2,1].set_ylabel('Age at Hire')
    
    # Color the boxplots
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 9. NFL Coordinator experience vs Age (t-SNE space overlay)
    nfl_coord_exp = data['Feature 7'].values
    ages = data['Feature 1'].values
    
    # Create size mapping for NFL coordinator experience
    sizes = 20 + (nfl_coord_exp - nfl_coord_exp.min()) / (nfl_coord_exp.max() - nfl_coord_exp.min()) * 100
    
    scatter3 = axes[2,2].scatter(tsne_features[:, 0], tsne_features[:, 1], 
                               c=ages, s=sizes, alpha=0.6, cmap='viridis', 
                               edgecolors='black', linewidth=0.3)
    axes[2,2].set_title('Coach Profiles in t-SNE Space\n(Color=Age, Size=NFL Coord Exp)', fontsize=11, fontweight='bold')
    axes[2,2].set_xlabel('t-SNE 1')
    axes[2,2].set_ylabel('t-SNE 2')
    
    # Add colorbars
    cbar = plt.colorbar(scatter3, ax=axes[2,2])
    cbar.set_label('Age at Hire')
    
    # Adjust layout with more padding for 4K display
    plt.tight_layout(pad=3.0, h_pad=3.5, w_pad=2.0)
    plt.savefig('balanced_coach_clustering_final.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Return t-SNE coordinates for CSV export
    return tsne_features, tsne_3d_features

def export_clustering_data(data, labels, tsne_2d, tsne_3d):
    """Export clustering results with t-SNE coordinates to CSV for interactive visualization"""
    
    print("Exporting clustering data for interactive visualization...")
    
    # Create comprehensive dataset
    export_data = data.copy()
    
    # Add clustering results
    export_data['Cluster'] = labels
    
    # Add 2D t-SNE coordinates
    export_data['tSNE_X'] = tsne_2d[:, 0]
    export_data['tSNE_Y'] = tsne_2d[:, 1]
    
    # Add 3D t-SNE coordinates
    export_data['tSNE_X_3D'] = tsne_3d[:, 0]
    export_data['tSNE_Y_3D'] = tsne_3d[:, 1]
    export_data['tSNE_Z_3D'] = tsne_3d[:, 2]
    
    # Add cluster labels with descriptive names
    cluster_names = {
        0: 'Modern Coordinators',
        1: 'NFL Veterans', 
        2: 'High Performers',
        3: 'Position Coach Veterans',
        4: 'Veteran Retreads'
    }
    
    export_data['Cluster_Name'] = export_data['Cluster'].map(cluster_names)
    
    # Select key columns for the export
    key_columns = ['Coach Name', 'Year', 'Cluster', 'Cluster_Name',
                   'tSNE_X', 'tSNE_Y', 'tSNE_X_3D', 'tSNE_Y_3D', 'tSNE_Z_3D',
                   'Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 
                   'Feature 5', 'Feature 6', 'Feature 7', 'Feature 8',
                   'Avg 2Y Win Pct', 'Coach Tenure Class']
    
    # Only include columns that exist in the data
    available_columns = [col for col in key_columns if col in export_data.columns]
    export_subset = export_data[available_columns].copy()
    
    # Add readable column names
    column_rename = {
        'Feature 1': 'Age_at_Hire',
        'Feature 2': 'Previous_HC_Stints', 
        'Feature 3': 'College_Position_Years',
        'Feature 4': 'College_Coordinator_Years',
        'Feature 5': 'College_HC_Years',
        'Feature 6': 'NFL_Position_Years',
        'Feature 7': 'NFL_Coordinator_Years', 
        'Feature 8': 'NFL_HC_Years',
        'Avg 2Y Win Pct': 'Avg_2Y_Win_Pct',
        'Coach Tenure Class': 'Tenure_Class'
    }
    
    export_subset = export_subset.rename(columns=column_rename)
    
    # Save to CSV
    filename = 'coach_clustering_tsne_data.csv'
    export_subset.to_csv(filename, index=False)
    
    print(f"Data exported to: {filename}")
    print(f"Rows: {len(export_subset)}, Columns: {len(export_subset.columns)}")
    print(f"Ready for interactive 3D t-SNE visualization!")
    
    return filename

def main():
    """Main execution for comprehensive balanced clustering"""
    print("COMPREHENSIVE NFL HEAD COACH CLUSTERING")
    print("="*60)
    print("Focus: Balanced clusters using ALL features (pathway + performance)")
    print("Era: 1970+ for optimal data quality")
    print("="*60)
    
    # Load data
    df = pd.read_csv('master_data.csv')
    coach_data = df.copy()
    
    print(f"Loaded {len(coach_data)} coaching instances")
    
    # Apply balanced clustering
    labels, features, feature_names, modern_data = force_balanced_clustering(coach_data, n_clusters=5)
    
    # Analyze results
    analyze_balanced_clusters(modern_data, labels, features, feature_names)
    
    # Create visualizations
    tsne_2d, tsne_3d = create_balanced_visualization(modern_data, labels, features, feature_names)
    
    # Export data for interactive visualization
    export_clustering_data(modern_data, labels, tsne_2d, tsne_3d)
    
    print(f"\n{'='*60}")
    print("COMPREHENSIVE CLUSTERING COMPLETE!")
    print(f"{'='*60}")
    print("Key features:")
    print("- Uses ALL available features (pathway + performance)")
    print("- Modern era focus (1970+) for better data quality")
    print("- PCA dimensionality reduction for high-dimensional data")
    print("- Forced cluster balance through optimization")
    print("- Multiple trials to find best solution")
    print("- Comprehensive cluster profiling and visualization")
    print("\nVisualization saved: balanced_coach_clustering_final.png")

if __name__ == "__main__":
    main()