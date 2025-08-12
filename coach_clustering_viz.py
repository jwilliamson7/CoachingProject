import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare head coach data for clustering"""
    df = pd.read_csv('master_data.csv')
    
    # Filter for head coach hires only (exclude tenure class -1 for cleaner viz)
    coach_data = df[df['Coach Tenure Class'] != -1].copy()
    
    # Get feature columns (Features 1-150)
    feature_cols = [f'Feature {i}' for i in range(1, 151)]
    
    # Remove any rows with all NaN features
    coach_data = coach_data.dropna(subset=feature_cols, how='all')
    
    # Fill remaining NaNs with 0
    coach_data[feature_cols] = coach_data[feature_cols].fillna(0)
    
    print(f"Loaded {len(coach_data)} head coach hiring instances")
    print(f"Tenure distribution:")
    print(coach_data['Coach Tenure Class'].value_counts().sort_index())
    
    return coach_data, feature_cols

def perform_clustering(data, features, n_clusters=5):
    """Perform K-means clustering on the data"""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[features])
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    # DBSCAN for comparison
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(scaled_features)
    
    return cluster_labels, dbscan_labels, scaled_features, scaler

def reduce_dimensions(scaled_features):
    """Reduce dimensions for visualization"""
    # PCA
    pca = PCA(n_components=2, random_state=42)
    pca_features = pca.fit_transform(scaled_features)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_features = tsne.fit_transform(scaled_features)
    
    return pca_features, tsne_features, pca

def create_visualizations(data, cluster_labels, dbscan_labels, pca_features, tsne_features, pca):
    """Create comprehensive clustering visualizations"""
    
    # Set up the plot style
    plt.style.use('default')
    fig = plt.figure(figsize=(16, 12))
    
    # Color maps
    tenure_colors = {0: '#FF6B6B', 1: '#4ECDC4', 2: '#45B7D1'}
    tenure_labels = {0: '≤2 years', 1: '3-4 years', 2: '5+ years'}
    
    # 1. K-means clustering with PCA
    plt.subplot(3, 3, 1)
    scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], 
                         c=cluster_labels, cmap='viridis', alpha=0.7, s=50)
    plt.title('K-Means Clustering (PCA)', fontsize=14, fontweight='bold')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.colorbar(scatter, label='Cluster')
    
    # 2. K-means clustering with t-SNE
    plt.subplot(3, 3, 2)
    scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], 
                         c=cluster_labels, cmap='viridis', alpha=0.7, s=50)
    plt.title('K-Means Clustering (t-SNE)', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.colorbar(scatter, label='Cluster')
    
    # 3. DBSCAN clustering with PCA
    plt.subplot(3, 3, 3)
    unique_labels = np.unique(dbscan_labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    for label, color in zip(unique_labels, colors):
        mask = dbscan_labels == label
        if label == -1:
            plt.scatter(pca_features[mask, 0], pca_features[mask, 1], 
                       c='black', marker='x', s=50, alpha=0.7, label='Noise')
        else:
            plt.scatter(pca_features[mask, 0], pca_features[mask, 1], 
                       c=[color], s=50, alpha=0.7, label=f'Cluster {label}')
    plt.title('DBSCAN Clustering (PCA)', fontsize=14, fontweight='bold')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. Actual tenure classes with PCA
    plt.subplot(3, 3, 4)
    for tenure_class in [0, 1, 2]:
        mask = data['Coach Tenure Class'] == tenure_class
        plt.scatter(pca_features[mask, 0], pca_features[mask, 1], 
                   c=tenure_colors[tenure_class], s=50, alpha=0.7, 
                   label=tenure_labels[tenure_class])
    plt.title('Actual Tenure Classes (PCA)', fontsize=14, fontweight='bold')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.legend()
    
    # 5. Actual tenure classes with t-SNE
    plt.subplot(3, 3, 5)
    for tenure_class in [0, 1, 2]:
        mask = data['Coach Tenure Class'] == tenure_class
        plt.scatter(tsne_features[mask, 0], tsne_features[mask, 1], 
                   c=tenure_colors[tenure_class], s=50, alpha=0.7, 
                   label=tenure_labels[tenure_class])
    plt.title('Actual Tenure Classes (t-SNE)', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    
    # 6. Hire year timeline
    plt.subplot(3, 3, 6)
    scatter = plt.scatter(data['Year'], pca_features[:, 0], 
                         c=data['Coach Tenure Class'], cmap='RdYlBu', s=50, alpha=0.7)
    plt.title('Coach Hires Over Time (PC1)', fontsize=14, fontweight='bold')
    plt.xlabel('Hire Year')
    plt.ylabel('PC1')
    plt.colorbar(scatter, label='Tenure Class')
    
    # 7. Age at hire vs tenure
    plt.subplot(3, 3, 7)
    age_feature = data['Feature 1']  # Assuming Feature 1 is age
    for tenure_class in [0, 1, 2]:
        mask = data['Coach Tenure Class'] == tenure_class
        plt.scatter(age_feature[mask], data['Avg 2Y Win Pct'][mask], 
                   c=tenure_colors[tenure_class], s=50, alpha=0.7, 
                   label=tenure_labels[tenure_class])
    plt.title('Age vs 2-Year Win %', fontsize=14, fontweight='bold')
    plt.xlabel('Age at Hire')
    plt.ylabel('2-Year Win Percentage')
    plt.legend()
    
    # 8. Cluster characteristics heatmap
    plt.subplot(3, 3, 8)
    cluster_means = []
    important_features = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5', 
                         'Feature 6', 'Feature 7', 'Feature 8']  # First 8 features
    
    for cluster in range(max(cluster_labels) + 1):
        mask = cluster_labels == cluster
        cluster_mean = data[mask][important_features].mean()
        cluster_means.append(cluster_mean)
    
    cluster_df = pd.DataFrame(cluster_means)
    sns.heatmap(cluster_df.T, annot=True, fmt='.2f', cmap='coolwarm', 
                xticklabels=[f'Cluster {i}' for i in range(len(cluster_means))],
                yticklabels=[f'F{i+1}' for i in range(len(important_features))])
    plt.title('Cluster Feature Profiles', fontsize=14, fontweight='bold')
    
    # 9. Decade analysis
    plt.subplot(3, 3, 9)
    data['Decade'] = (data['Year'] // 10) * 10
    decade_tenure = data.groupby(['Decade', 'Coach Tenure Class']).size().unstack(fill_value=0)
    decade_tenure_pct = decade_tenure.div(decade_tenure.sum(axis=1), axis=0) * 100
    
    decade_tenure_pct.plot(kind='bar', stacked=True, 
                          color=[tenure_colors[i] for i in [0, 1, 2]], ax=plt.gca())
    plt.title('Tenure Distribution by Decade', fontsize=14, fontweight='bold')
    plt.xlabel('Decade')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)
    plt.legend(title='Tenure Class', labels=[tenure_labels[i] for i in [0, 1, 2]])
    
    plt.tight_layout()
    plt.savefig('coach_clustering_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_clusters(data, cluster_labels, features):
    """Analyze cluster characteristics"""
    print("\n" + "="*60)
    print("CLUSTER ANALYSIS")
    print("="*60)
    
    for cluster in range(max(cluster_labels) + 1):
        mask = cluster_labels == cluster
        cluster_data = data[mask]
        
        print(f"\nCluster {cluster} ({len(cluster_data)} coaches):")
        print(f"  Avg Age at Hire: {cluster_data['Feature 1'].mean():.1f}")
        print(f"  Avg 2Y Win %: {cluster_data['Avg 2Y Win Pct'].mean():.3f}")
        print(f"  Tenure Distribution:")
        tenure_dist = cluster_data['Coach Tenure Class'].value_counts().sort_index()
        for tenure, count in tenure_dist.items():
            pct = count / len(cluster_data) * 100
            print(f"    {tenure} ({['≤2yr', '3-4yr', '5+yr'][tenure]}): {count} coaches ({pct:.1f}%)")
        
        # Most common coaches in this cluster
        top_coaches = cluster_data.nlargest(3, 'Avg 2Y Win Pct')[['Coach Name', 'Year', 'Avg 2Y Win Pct']]
        print(f"  Top Performers:")
        for _, coach in top_coaches.iterrows():
            print(f"    {coach['Coach Name']} ({coach['Year']}) - {coach['Avg 2Y Win Pct']:.3f}")

def main():
    """Main execution function"""
    print("NFL Head Coach Clustering Analysis")
    print("="*50)
    
    # Load and prepare data
    data, features = load_and_prepare_data()
    
    # Perform clustering
    cluster_labels, dbscan_labels, scaled_features, scaler = perform_clustering(data, features)
    
    # Reduce dimensions
    pca_features, tsne_features, pca = reduce_dimensions(scaled_features)
    
    # Create visualizations
    create_visualizations(data, cluster_labels, dbscan_labels, pca_features, tsne_features, pca)
    
    # Analyze clusters
    analyze_clusters(data, cluster_labels, features)
    
    print(f"\nVisualization saved as 'coach_clustering_analysis.png'")
    print("Analysis complete!")

if __name__ == "__main__":
    main()