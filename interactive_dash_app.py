import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, callback
import numpy as np

# Load the data
def load_data():
    try:
        df = pd.read_csv('coach_clustering_tsne_data.csv')
        return df
    except FileNotFoundError:
        print("ERROR: coach_clustering_tsne_data.csv not found!")
        return None

# Initialize the Dash app
app = dash.Dash(__name__)

# Load data
df = load_data()
if df is None:
    print("Please run balanced_clustering.py first to generate the data.")
    exit()

# Create custom color palette for clusters
cluster_colors = {
    'Modern Coordinators': '#1f77b4',
    'NFL Veterans': '#ff7f0e',
    'High Performers': '#2ca02c', 
    'Position Coach Veterans': '#d62728',
    'Veteran Retreads': '#9467bd'
}

# Get unique coaches for search dropdown
coach_options = [{'label': coach, 'value': coach} for coach in sorted(df['Coach Name'].unique())]

# Define the layout
app.layout = html.Div([
    html.H1("Interactive 3D Coach Clustering Visualization", 
            style={'textAlign': 'center', 'marginBottom': 30}),
    
    # Controls row
    html.Div([
        # Year range slider
        html.Div([
            html.Label("Year Range:", style={'fontWeight': 'bold'}),
            dcc.RangeSlider(
                id='year-range-slider',
                min=int(df['Year'].min()),
                max=int(df['Year'].max()),
                value=[int(df['Year'].min()), int(df['Year'].max())],
                marks={year: str(year) for year in range(int(df['Year'].min()), int(df['Year'].max())+1, 5)},
                step=1,
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
        
        # Coach search dropdown
        html.Div([
            html.Label("Search Coaches:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='coach-search',
                options=coach_options,
                value=[],
                multi=True,
                placeholder="Type to search and select coaches...",
                style={'width': '100%'}
            )
        ], style={'width': '48%', 'display': 'inline-block'})
    ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
    
    # Cluster filter checkboxes
    html.Div([
        html.Label("Show Clusters:", style={'fontWeight': 'bold', 'marginRight': '20px'}),
        dcc.Checklist(
            id='cluster-filter',
            options=[{'label': cluster, 'value': cluster} for cluster in df['Cluster_Name'].unique()],
            value=list(df['Cluster_Name'].unique()),
            inline=True,
            style={'marginTop': '10px'}
        )
    ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
    
    # 3D plot
    dcc.Graph(
        id='3d-scatter-plot',
        style={'height': '700px'}
    ),
    
    # Info panel
    html.Div(id='info-panel', style={
        'margin': '20px',
        'padding': '20px', 
        'backgroundColor': '#e9ecef',
        'borderRadius': '5px'
    })
])

# Callback for updating the 3D plot
@app.callback(
    [Output('3d-scatter-plot', 'figure'),
     Output('info-panel', 'children')],
    [Input('year-range-slider', 'value'),
     Input('coach-search', 'value'),
     Input('cluster-filter', 'value')]
)
def update_plot(year_range, selected_coaches, selected_clusters):
    # Filter data based on inputs
    filtered_df = df.copy()
    
    # Filter by year range
    filtered_df = filtered_df[(filtered_df['Year'] >= year_range[0]) & 
                             (filtered_df['Year'] <= year_range[1])]
    
    # Filter by selected clusters
    filtered_df = filtered_df[filtered_df['Cluster_Name'].isin(selected_clusters)]
    
    # Create the 3D plot
    fig = go.Figure()
    
    # Add traces for each cluster
    for cluster_name in selected_clusters:
        cluster_data = filtered_df[filtered_df['Cluster_Name'] == cluster_name]
        
        if len(cluster_data) > 0:
            # Highlight searched coaches
            if selected_coaches:
                highlighted = cluster_data[cluster_data['Coach Name'].isin(selected_coaches)]
                normal = cluster_data[~cluster_data['Coach Name'].isin(selected_coaches)]
                
                # Add normal coaches with smaller, more transparent markers
                if len(normal) > 0:
                    fig.add_trace(go.Scatter3d(
                        x=normal['tSNE_X_3D'],
                        y=normal['tSNE_Y_3D'],
                        z=normal['tSNE_Z_3D'],
                        mode='markers',
                        marker=dict(
                            size=normal['Age_at_Hire'] * 0.25,
                            color=cluster_colors.get(cluster_name, '#1f77b4'),
                            opacity=0.4,
                            line=dict(width=0.5, color='black')
                        ),
                        name=cluster_name,
                        text=normal['Coach Name'],
                        customdata=normal[['Coach Name', 'Year', 'Age_at_Hire', 'NFL_Coordinator_Years', 
                                         'NFL_Position_Years', 'College_HC_Years', 'Previous_HC_Stints']].values,
                        hovertemplate='<b>%{customdata[0]}</b> (%{customdata[1]})<br>' +
                                    'Cluster: ' + cluster_name + '<br>' +
                                    'Age: %{customdata[2]}<br>' +
                                    'NFL Coordinator: %{customdata[3]} years<br>' +
                                    'NFL Position: %{customdata[4]} years<br>' +
                                    'College HC: %{customdata[5]} years<br>' +
                                    'Previous HC Jobs: %{customdata[6]}<extra></extra>',
                        showlegend=True
                    ))
                
                # Add highlighted coaches with larger, bright markers
                if len(highlighted) > 0:
                    fig.add_trace(go.Scatter3d(
                        x=highlighted['tSNE_X_3D'],
                        y=highlighted['tSNE_Y_3D'],
                        z=highlighted['tSNE_Z_3D'],
                        mode='markers',
                        marker=dict(
                            size=highlighted['Age_at_Hire'] * 0.5,
                            color='gold',
                            opacity=1.0,
                            line=dict(width=3, color='red')
                        ),
                        name=f'{cluster_name} (Selected)',
                        text=highlighted['Coach Name'],
                        customdata=highlighted[['Coach Name', 'Year', 'Age_at_Hire', 'NFL_Coordinator_Years',
                                              'NFL_Position_Years', 'College_HC_Years', 'Previous_HC_Stints']].values,
                        hovertemplate='<b>%{customdata[0]}</b> (%{customdata[1]})<br>' +
                                    'Cluster: ' + cluster_name + '<br>' +
                                    'Age: %{customdata[2]}<br>' +
                                    'NFL Coordinator: %{customdata[3]} years<br>' +
                                    'NFL Position: %{customdata[4]} years<br>' +
                                    'College HC: %{customdata[5]} years<br>' +
                                    'Previous HC Jobs: %{customdata[6]}<extra></extra>',
                        showlegend=True
                    ))
            else:
                # No coaches selected, show all normally
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
                    customdata=cluster_data[['Coach Name', 'Year', 'Age_at_Hire', 'NFL_Coordinator_Years',
                                           'NFL_Position_Years', 'College_HC_Years', 'Previous_HC_Stints']].values,
                    hovertemplate='<b>%{customdata[0]}</b> (%{customdata[1]})<br>' +
                                'Cluster: ' + cluster_name + '<br>' +
                                'Age: %{customdata[2]}<br>' +
                                'NFL Coordinator: %{customdata[3]} years<br>' +
                                'NFL Position: %{customdata[4]} years<br>' +
                                'College HC: %{customdata[5]} years<br>' +
                                'Previous HC Jobs: %{customdata[6]}<extra></extra>',
                    showlegend=True
                ))
    
    # Update layout
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
        title=dict(
            text="3D Coach Clustering with Interactive Filters",
            x=0.5,
            font=dict(size=16)
        ),
        font=dict(size=12),
        legend=dict(
            yanchor="top",
            y=1,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        height=700,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    # Create info panel content
    total_coaches = len(filtered_df)
    info_content = [
        html.H4("Dataset Summary"),
        html.P(f"Showing {total_coaches} coaches from {year_range[0]} to {year_range[1]}"),
        html.P(f"Clusters displayed: {', '.join(selected_clusters)}"),
    ]
    
    if selected_coaches:
        info_content.append(html.P(f"Highlighted coaches: {', '.join(selected_coaches)}"))
    
    # Add cluster breakdown
    cluster_summary = filtered_df.groupby('Cluster_Name').size().to_dict()
    info_content.append(html.H5("Cluster Distribution:"))
    for cluster, count in cluster_summary.items():
        pct = (count / total_coaches * 100) if total_coaches > 0 else 0
        info_content.append(html.P(f"• {cluster}: {count} coaches ({pct:.1f}%)"))
    
    return fig, info_content

if __name__ == '__main__':
    print("Starting Interactive Coach Clustering App...")
    print("=" * 50)
    print("Open your browser to one of these URLs:")
    print("  http://127.0.0.1:8050/")
    print("  http://localhost:8050/")
    print("=" * 50)
    try:
        app.run(debug=False, host='127.0.0.1', port=8050)
    except Exception as e:
        print(f"Error starting server: {e}")
        print("Trying alternative port...")
        app.run(debug=False, host='127.0.0.1', port=8051)