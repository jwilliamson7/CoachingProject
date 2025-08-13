import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import numpy as np

def load_clustering_data():
    """Load the t-SNE clustering data"""
    
    try:
        df = pd.read_csv('coach_clustering_tsne_data.csv')
        print(f"Loaded {len(df)} coaches with clustering data")
        return df
    except FileNotFoundError:
        print("ERROR: coach_clustering_tsne_data.csv not found!")
        print("Please run balanced_clustering.py first to generate the data.")
        return None

def create_3d_scatter_plot(df):
    """Create the main 3D scatter plot with filters and search"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create custom color palette for clusters
    cluster_colors = {
        'Modern Coordinators': '#1f77b4',      # Blue
        'NFL Veterans': '#ff7f0e',             # Orange  
        'High Performers': '#2ca02c',          # Green
        'Position Coach Veterans': '#d62728',   # Red
        'Veteran Retreads': '#9467bd'          # Purple
    }
    
    # Create hover text with detailed coach info
    hover_text = []
    for _, row in df.iterrows():
        hover_info = (
            f"<b>{row['Coach Name']}</b> ({row['Year']})<br>"
            f"Cluster: {row['Cluster_Name']}<br>"
            f"Age: {row['Age_at_Hire']}<br>"
            f"NFL Coordinator: {row['NFL_Coordinator_Years']} years<br>"
            f"NFL Position: {row['NFL_Position_Years']} years<br>"
            f"College HC: {row['College_HC_Years']} years<br>"
            f"Previous HC Jobs: {row['Previous_HC_Stints']}<br>"
            f"Win %: {row['Avg_2Y_Win_Pct']:.3f}" if pd.notna(row['Avg_2Y_Win_Pct']) else "Win %: N/A"
        )
        hover_text.append(hover_info)
    
    df['hover_text'] = hover_text
    
    # Create base figure using graph_objects for more control
    fig = go.Figure()
    
    # Add traces for each cluster
    for cluster_name in df['Cluster_Name'].unique():
        cluster_data = df[df['Cluster_Name'] == cluster_name]
        
        fig.add_trace(go.Scatter3d(
            x=cluster_data['tSNE_X_3D'],
            y=cluster_data['tSNE_Y_3D'], 
            z=cluster_data['tSNE_Z_3D'],
            mode='markers',
            marker=dict(
                size=cluster_data['Age_at_Hire'] * 0.3,
                color=cluster_colors.get(cluster_name, '#1f77b4'),
                opacity=0.8,
                line=dict(width=0.5, color='black')
            ),
            name=cluster_name,
            text=cluster_data['Coach Name'],
            customdata=cluster_data[['Coach Name', 'Year', 'hover_text']].values,
            hovertemplate='%{customdata[2]}<extra></extra>'
        ))
    
    # Add year range slider and coach search controls
    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())
    
    # Update layout with controls
    fig.update_layout(
        scene=dict(
            xaxis_title='t-SNE Dimension 1',
            yaxis_title='t-SNE Dimension 2', 
            zaxis_title='t-SNE Dimension 3',
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='lightgray', gridwidth=1),
            yaxis=dict(gridcolor='lightgray', gridwidth=1),
            zaxis=dict(gridcolor='lightgray', gridwidth=1)
        ),
        font=dict(size=12),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left", 
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        title=dict(
            text="<b>Interactive 3D Coach Clustering Visualization</b><br><sub>Based on Career Pathway + Performance Features (t-SNE)</sub>",
            x=0.5
        ),
        width=1200,
        height=900,
        margin=dict(l=0, r=50, t=100, b=100),
        
        # Add range slider for year filtering
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"visible": [True] * len(fig.data)}],
                        label="Show All",
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": [True] * len(fig.data)}],
                        label="Reset Filters", 
                        method="restyle"
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.08,
                yanchor="top"
            )
        ],
        
        # Add sliders for year range
        sliders=[
            dict(
                active=max_year - min_year,
                currentvalue={"prefix": "Year Range: "},
                pad={"t": 50},
                steps=[
                    dict(
                        args=[
                            dict(
                                transforms=[
                                    dict(
                                        type='filter',
                                        target='customdata[1]',
                                        operation='>=',
                                        value=year
                                    )
                                ]
                            )
                        ],
                        label=str(year),
                        method='restyle'
                    ) for year in range(min_year, max_year + 1, 5)
                ],
                x=0.1,
                len=0.8,
                y=0,
                yanchor="top"
            )
        ]
    )
    
    return fig

def create_cluster_analysis_plots(df):
    """Create additional analysis plots"""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Cluster Size Distribution', 'Age Distribution by Cluster', 
            'NFL Coordinator Experience', 'Win % by Cluster', 
            'Hiring Timeline', 'Experience Profile Comparison'
        ),
        specs=[[{'type': 'bar'}, {'type': 'box'}, {'type': 'violin'}],
               [{'type': 'box'}, {'type': 'scatter'}, {'type': 'bar'}]]
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 1. Cluster sizes
    cluster_counts = df['Cluster_Name'].value_counts()
    fig.add_trace(
        go.Bar(x=cluster_counts.index, y=cluster_counts.values, 
               marker_color=colors, name='Cluster Size'),
        row=1, col=1
    )
    
    # 2. Age distribution by cluster
    for i, cluster in enumerate(df['Cluster_Name'].unique()):
        cluster_data = df[df['Cluster_Name'] == cluster]
        fig.add_trace(
            go.Box(y=cluster_data['Age_at_Hire'], name=cluster,
                   marker_color=colors[i], showlegend=False),
            row=1, col=2
        )
    
    # 3. NFL Coordinator experience
    for i, cluster in enumerate(df['Cluster_Name'].unique()):
        cluster_data = df[df['Cluster_Name'] == cluster]
        fig.add_trace(
            go.Violin(y=cluster_data['NFL_Coordinator_Years'], name=cluster,
                     line_color=colors[i], showlegend=False),
            row=1, col=3
        )
    
    # 4. Win % by cluster (known outcomes only)
    known_outcomes = df[df['Tenure_Class'] != -1]
    for i, cluster in enumerate(known_outcomes['Cluster_Name'].unique()):
        cluster_data = known_outcomes[known_outcomes['Cluster_Name'] == cluster]
        fig.add_trace(
            go.Box(y=cluster_data['Avg_2Y_Win_Pct'], name=cluster,
                   marker_color=colors[i], showlegend=False),
            row=2, col=1
        )
    
    # 5. Hiring timeline
    fig.add_trace(
        go.Scatter(x=df['Year'], y=df['Age_at_Hire'], 
                   mode='markers', marker=dict(color=df['Cluster'], 
                   colorscale='Viridis', size=5),
                   text=df['Coach Name'], name='Hires Over Time'),
        row=2, col=2
    )
    
    # 6. Average experience by cluster
    exp_cols = ['College_Position_Years', 'College_Coordinator_Years', 'College_HC_Years',
                'NFL_Position_Years', 'NFL_Coordinator_Years', 'NFL_HC_Years']
    
    cluster_means = df.groupby('Cluster_Name')[exp_cols].mean()
    total_exp = cluster_means.sum(axis=1)
    
    fig.add_trace(
        go.Bar(x=cluster_means.index, y=total_exp.values,
               marker_color=colors, name='Total Experience'),
        row=2, col=3
    )
    
    # Update layout
    fig.update_layout(
        title="<b>Coach Clustering Analysis Dashboard</b>",
        height=800,
        showlegend=False,
        font=dict(size=10)
    )
    
    return fig

def create_animated_timeline(df):
    """Create an animated timeline showing coaching hires over time in 5-year increments"""
    
    # Create 5-year periods for animation
    df_timeline = df.copy()
    df_timeline['Year_Period'] = ((df_timeline['Year'] // 5) * 5).astype(str) + '-' + ((df_timeline['Year'] // 5) * 5 + 4).astype(str)
    
    # Sort periods chronologically
    unique_periods = sorted(df_timeline['Year_Period'].unique())
    
    print(f"Timeline periods: {unique_periods}")
    
    # Create animated scatter plot
    fig = px.scatter_3d(
        df_timeline, 
        x='tSNE_X_3D', 
        y='tSNE_Y_3D', 
        z='tSNE_Z_3D',
        color='Cluster_Name',
        size='Age_at_Hire',
        animation_frame='Year_Period',
        hover_name='Coach Name',
        title="<b>Coach Hiring Animation Over Time</b><br><sub>3D t-SNE Clustering by 5-Year Periods</sub>",
        size_max=15,
        category_orders={'Year_Period': unique_periods}  # Ensure chronological order
    )
    
    # Get the full range of t-SNE coordinates to fix axes
    x_range = [df_timeline['tSNE_X_3D'].min(), df_timeline['tSNE_X_3D'].max()]
    y_range = [df_timeline['tSNE_Y_3D'].min(), df_timeline['tSNE_Y_3D'].max()]
    z_range = [df_timeline['tSNE_Z_3D'].min(), df_timeline['tSNE_Z_3D'].max()]
    
    # Update layout with fixed axis ranges and aspect ratios
    fig.update_layout(
        scene=dict(
            xaxis_title='t-SNE Dimension 1',
            yaxis_title='t-SNE Dimension 2',
            zaxis_title='t-SNE Dimension 3',
            xaxis=dict(range=x_range),
            yaxis=dict(range=y_range),
            zaxis=dict(range=z_range),
            aspectmode='cube'  # Forces equal aspect ratios
        ),
        width=1200,
        height=800
    )
    
    return fig

def create_interactive_dashboard():
    """Create the complete interactive dashboard"""
    
    print("Creating Interactive 3D Coach Clustering Visualization...")
    print("=" * 60)
    
    # Load data
    df = load_clustering_data()
    if df is None:
        return
    
    print(f"Data loaded successfully!")
    print(f"Coaches: {len(df)}, Clusters: {df['Cluster_Name'].nunique()}")
    
    # Create main 3D plot
    print("Creating main 3D scatter plot...")
    main_fig = create_3d_scatter_plot(df)
    
    # Create analysis dashboard  
    print("Creating analysis dashboard...")
    analysis_fig = create_cluster_analysis_plots(df)
    
    # Create animated timeline
    print("Creating animated timeline...")
    timeline_fig = create_animated_timeline(df)
    
    # Save all visualizations
    print("\nSaving interactive visualizations...")
    
    # Main 3D plot
    main_fig.write_html("interactive_3d_coaching_clusters.html")
    print("* Main 3D plot saved: interactive_3d_coaching_clusters.html")
    
    # Analysis dashboard
    analysis_fig.write_html("coaching_cluster_analysis_dashboard.html") 
    print("* Analysis dashboard saved: coaching_cluster_analysis_dashboard.html")
    
    # Animated timeline
    timeline_fig.write_html("coaching_timeline_animation.html")
    print("* Timeline animation saved: coaching_timeline_animation.html")
    
    print("\n" + "=" * 60)
    print("INTERACTIVE VISUALIZATIONS COMPLETE!")
    print("=" * 60)
    print("Open these HTML files in your browser:")
    print("   - interactive_3d_coaching_clusters.html - Main 3D exploration")
    print("   - coaching_cluster_analysis_dashboard.html - Statistical analysis")  
    print("   - coaching_timeline_animation.html - Animated timeline")
    print("\nFeatures:")
    print("   - Rotate, zoom, pan in 3D space")
    print("   - Hover for detailed coach information")
    print("   - Click legend to show/hide clusters") 
    print("   - Timeline animation shows hiring patterns over time")
    
    # Display the main plot
    print("\nLaunching main 3D visualization...")
    main_fig.show()
    
    return main_fig, analysis_fig, timeline_fig

if __name__ == "__main__":
    create_interactive_dashboard()