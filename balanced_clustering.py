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
    """Force more balanced clustering by penalizing uneven cluster sizes"""
    
    # Use only core interpretable features
    core_features = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 
                    'Feature 5', 'Feature 6', 'Feature 7', 'Feature 8']
    
    # Clean and scale data
    clean_data = data[core_features].copy()
    for col in core_features:
        clean_data[col] = clean_data[col].fillna(clean_data[col].median())
    
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(clean_data)
    
    print(f"Forcing {n_clusters} balanced clusters on {len(core_features)} core features...")
    
    best_labels = None
    best_score = -1
    best_balance = 0
    
    # Try multiple random initializations to find balanced solution
    for trial in range(50):
        kmeans = KMeans(n_clusters=n_clusters, random_state=trial, n_init=20, init='k-means++')
        labels = kmeans.fit_predict(scaled_data)
        
        # Calculate cluster balance
        cluster_sizes = np.bincount(labels)
        balance_score = 1 - (np.std(cluster_sizes) / np.mean(cluster_sizes))
        balance_score = max(0, balance_score)
        
        # Calculate silhouette score
        sil_score = silhouette_score(scaled_data, labels)
        
        # Combined score that heavily weights balance
        combined_score = sil_score + (balance_penalty * balance_score)
        
        if combined_score > best_score:
            best_score = combined_score
            best_labels = labels.copy()
            best_balance = balance_score
            best_sil = sil_score
    
    print(f"Best solution - Silhouette: {best_sil:.3f}, Balance: {best_balance:.3f}, Combined: {best_score:.3f}")
    
    return best_labels, scaled_data, core_features

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
        print(f"  College Experience: {cluster_data['Feature 3'].mean():.1f} years")
        print(f"  NFL Experience: {cluster_data['Feature 4'].mean():.1f} years")
        print(f"  Position Coach: {cluster_data['Feature 5'].mean():.1f} years")
        print(f"  Coordinator: {cluster_data['Feature 6'].mean():.1f} years")
        print(f"  Previous HC: {cluster_data['Feature 7'].mean():.1f} years")
        print(f"  Multiple Levels: {cluster_data['Feature 8'].mean():.1f}")
        
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
        exp_mean = cluster_data['Feature 4'].mean()
        coord_mean = cluster_data['Feature 6'].mean()
        
        # Generate cluster description
        if age_mean < 40:
            age_desc = "Very young"
        elif age_mean < 45:
            age_desc = "Young"
        elif age_mean < 50:
            age_desc = "Middle-aged"
        else:
            age_desc = "Veteran"
            
        if exp_mean < 5:
            exp_desc = "limited NFL experience"
        elif exp_mean < 15:
            exp_desc = "moderate NFL experience"
        else:
            exp_desc = "extensive NFL experience"
            
        if coord_mean < 2:
            coord_desc = "minimal coordinator background"
        elif coord_mean < 5:
            coord_desc = "some coordinator experience"
        else:
            coord_desc = "strong coordinator background"
            
        print(f"\nCluster Profile: {age_desc} coaches with {exp_desc} and {coord_desc}")

def create_balanced_visualization(data, labels, features, feature_names):
    """Create visualizations for balanced clustering"""
    
    # Create PCA for visualization
    pca_2d = PCA(n_components=2, random_state=42)
    pca_features = pca_2d.fit_transform(features)
    
    # Create t-SNE for visualization
    print("Computing t-SNE visualization...")
    perplexity = min(30, len(features) // 8)  # Adjust perplexity based on data size
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                n_iter=500, learning_rate=200)
    tsne_features = tsne.fit_transform(features)
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    
    unique_labels = np.unique(labels)
    cluster_sizes = [np.sum(labels == label) for label in unique_labels]
    total_coaches = len(labels)
    
    # 1. PCA cluster plot
    scatter = axes[0,0].scatter(pca_features[:, 0], pca_features[:, 1], 
                              c=labels, cmap='Set3', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[0,0].set_title('Balanced Coach Clusters\n(PCA Visualization)', fontweight='bold')
    axes[0,0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
    axes[0,0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
    plt.colorbar(scatter, ax=axes[0,0], label='Cluster')
    
    # 2. t-SNE cluster plot
    scatter2 = axes[0,1].scatter(tsne_features[:, 0], tsne_features[:, 1], 
                               c=labels, cmap='Set3', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[0,1].set_title('Balanced Coach Clusters\n(t-SNE Visualization)', fontweight='bold')
    axes[0,1].set_xlabel('t-SNE 1')
    axes[0,1].set_ylabel('t-SNE 2')
    plt.colorbar(scatter2, ax=axes[0,1], label='Cluster')
    
    # 3. Cluster size distribution
    bars = axes[0,2].bar([f'C{i}' for i in unique_labels], cluster_sizes, 
                        color='skyblue', alpha=0.7, edgecolor='black')
    axes[0,2].set_title('Cluster Size Distribution', fontweight='bold')
    axes[0,2].set_ylabel('Number of Coaches')
    axes[0,2].set_xlabel('Cluster')
    
    # Add percentage labels
    for bar, size in zip(bars, cluster_sizes):
        height = bar.get_height()
        pct = (size / total_coaches) * 100
        axes[0,2].text(bar.get_x() + bar.get_width()/2., height + 2,
                      f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. Age vs Experience colored by cluster (PCA view)
    axes[1,0].scatter(data['Feature 1'], data['Feature 4'], c=labels, cmap='Set3', 
                     s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[1,0].set_title('Age vs NFL Experience\n(by Cluster)', fontweight='bold')
    axes[1,0].set_xlabel('Age at Hire')
    axes[1,0].set_ylabel('NFL Experience (Years)')
    
    # 5. Experience profile by cluster
    experience_features = ['Feature 3', 'Feature 4', 'Feature 5', 'Feature 6', 'Feature 7']
    exp_labels = ['College', 'NFL', 'Position', 'Coordinator', 'Head Coach']
    
    for cluster in unique_labels:
        mask = labels == cluster
        cluster_exp = data[mask][experience_features].mean()
        axes[1,1].plot(exp_labels, cluster_exp.values, 'o-', linewidth=2, 
                      markersize=8, label=f'Cluster {cluster}', alpha=0.8)
    
    axes[1,1].set_title('Experience Profiles by Cluster', fontweight='bold')
    axes[1,1].set_xlabel('Experience Type')
    axes[1,1].set_ylabel('Average Years')
    axes[1,1].legend()
    axes[1,1].tick_params(axis='x', rotation=45)
    
    # 6. Win percentage distribution by cluster (for known outcomes)
    known_outcomes = data['Coach Tenure Class'] != -1
    if known_outcomes.any():
        for cluster in unique_labels:
            mask = (labels == cluster) & known_outcomes
            if mask.any():
                win_pcts = data[mask]['Avg 2Y Win Pct']
                axes[1,2].hist(win_pcts, bins=10, alpha=0.6, label=f'Cluster {cluster}')
        
        axes[1,2].set_title('Win % Distribution by Cluster\n(Known Outcomes)', fontweight='bold')
        axes[1,2].set_xlabel('2-Year Win Percentage')
        axes[1,2].set_ylabel('Count')
        axes[1,2].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='0.500')
        axes[1,2].legend()
    
    # 7. Cluster characteristics heatmap
    cluster_means = []
    for cluster in unique_labels:
        mask = labels == cluster
        cluster_mean = data[mask][feature_names].mean()
        cluster_means.append(cluster_mean)
    
    cluster_df = pd.DataFrame(cluster_means, index=[f'Cluster {i}' for i in unique_labels])
    cluster_df_norm = (cluster_df - cluster_df.mean()) / cluster_df.std()  # Standardize for heatmap
    
    sns.heatmap(cluster_df_norm.T, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=axes[2,0], cbar_kws={'label': 'Standardized Value'})
    axes[2,0].set_title('Cluster Characteristics\n(Standardized)', fontweight='bold')
    axes[2,0].set_xlabel('Cluster')
    axes[2,0].set_ylabel('Feature')
    
    # Set y-axis labels for better readability
    feature_labels = ['Age', 'Prev HC', 'College', 'NFL', 'Position', 'Coord', 'HC Exp', 'Multi']
    axes[2,0].set_yticklabels(feature_labels, rotation=0)
    
    # 8. Age distribution by cluster
    age_data = [data[labels == i]['Feature 1'].values for i in unique_labels]
    bp = axes[2,1].boxplot(age_data, labels=[f'C{i}' for i in unique_labels], patch_artist=True)
    axes[2,1].set_title('Age Distribution by Cluster', fontweight='bold')
    axes[2,1].set_xlabel('Cluster')
    axes[2,1].set_ylabel('Age at Hire')
    
    # Color the boxplots
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 9. Coordinator experience vs Age (t-SNE space overlay)
    coord_exp = data['Feature 6'].values
    ages = data['Feature 1'].values
    
    # Create size mapping for coordinator experience
    sizes = 20 + (coord_exp - coord_exp.min()) / (coord_exp.max() - coord_exp.min()) * 100
    
    scatter3 = axes[2,2].scatter(tsne_features[:, 0], tsne_features[:, 1], 
                               c=ages, s=sizes, alpha=0.6, cmap='viridis', 
                               edgecolors='black', linewidth=0.3)
    axes[2,2].set_title('Coach Profiles in t-SNE Space\n(Color=Age, Size=Coord Exp)', fontweight='bold')
    axes[2,2].set_xlabel('t-SNE 1')
    axes[2,2].set_ylabel('t-SNE 2')
    
    # Add colorbars
    cbar = plt.colorbar(scatter3, ax=axes[2,2])
    cbar.set_label('Age at Hire')
    
    plt.tight_layout()
    plt.savefig('balanced_coach_clustering_final.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main execution for balanced clustering"""
    print("BALANCED NFL HEAD COACH CLUSTERING")
    print("="*50)
    print("Focus: Creating interpretable, balanced clusters")
    print("="*50)
    
    # Load data
    df = pd.read_csv('master_data.csv')
    coach_data = df.copy()
    
    print(f"Loaded {len(coach_data)} coaching instances")
    
    # Apply balanced clustering
    labels, features, feature_names = force_balanced_clustering(coach_data, n_clusters=5)
    
    # Analyze results
    analyze_balanced_clusters(coach_data, labels, features, feature_names)
    
    # Create visualizations
    create_balanced_visualization(coach_data, labels, features, feature_names)
    
    print(f"\n{'='*50}")
    print("BALANCED CLUSTERING COMPLETE!")
    print(f"{'='*50}")
    print("Key features:")
    print("- Forced cluster balance through optimization")
    print("- Focus on interpretable core features")
    print("- Multiple trials to find best solution")
    print("- Detailed cluster profiling")
    print("\nVisualization saved: balanced_coach_clustering_final.png")

if __name__ == "__main__":
    main()