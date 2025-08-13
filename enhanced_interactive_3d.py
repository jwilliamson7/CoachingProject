import pandas as pd
import plotly.graph_objects as go
import json

def create_enhanced_interactive_plot():
    """Create an enhanced HTML file with JavaScript-based filtering"""
    
    # Load data
    try:
        df = pd.read_csv('coach_clustering_tsne_data.csv')
        print(f"Loaded {len(df)} coaches")
    except FileNotFoundError:
        print("ERROR: coach_clustering_tsne_data.csv not found!")
        return
    
    # Create custom color palette
    cluster_colors = {
        'Modern Coordinators': '#1f77b4',
        'NFL Veterans': '#ff7f0e',
        'High Performers': '#2ca02c',
        'Position Coach Veterans': '#d62728',
        'Veteran Retreads': '#9467bd'
    }
    
    # Create base figure
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
            customdata=cluster_data[['Coach Name', 'Year', 'Age_at_Hire', 'NFL_Coordinator_Years',
                                   'NFL_Position_Years', 'College_HC_Years', 'Previous_HC_Stints']].values,
            hovertemplate='<b>%{customdata[0]}</b> (%{customdata[1]})<br>' +
                        'Cluster: ' + cluster_name + '<br>' +
                        'Age: %{customdata[2]}<br>' +
                        'NFL Coordinator: %{customdata[3]} years<br>' +
                        'NFL Position: %{customdata[4]} years<br>' +
                        'College HC: %{customdata[5]} years<br>' +
                        'Previous HC Jobs: %{customdata[6]}<extra></extra>'
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
            text="<b>Enhanced Interactive 3D Coach Clustering</b><br><sub>Use controls below to filter data</sub>",
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
        margin=dict(l=0, r=0, t=80, b=120)
    )
    
    # Get coach names for search functionality
    coach_names = sorted(df['Coach Name'].unique())
    
    # Create HTML with embedded controls
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced 3D Coach Clustering</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
            }}
            .controls {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
                border: 1px solid #dee2e6;
            }}
            .control-row {{
                display: flex;
                gap: 30px;
                margin-bottom: 15px;
                align-items: center;
            }}
            .control-group {{
                flex: 1;
            }}
            label {{
                font-weight: bold;
                display: block;
                margin-bottom: 5px;
            }}
            input[type="range"] {{
                width: 100%;
            }}
            input[type="text"] {{
                width: 100%;
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }}
            .search-results {{
                max-height: 150px;
                overflow-y: auto;
                border: 1px solid #ccc;
                border-top: none;
                background: white;
                display: none;
            }}
            .search-item {{
                padding: 8px;
                cursor: pointer;
                border-bottom: 1px solid #eee;
            }}
            .search-item:hover {{
                background-color: #f0f0f0;
            }}
            .selected-coaches {{
                display: flex;
                flex-wrap: wrap;
                gap: 5px;
                margin-top: 10px;
            }}
            .coach-tag {{
                background-color: #007bff;
                color: white;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 12px;
                cursor: pointer;
            }}
            .cluster-checkboxes {{
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
            }}
            button {{
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
                margin-right: 10px;
            }}
            button:hover {{
                background-color: #0056b3;
            }}
            .info-panel {{
                background-color: #e9ecef;
                padding: 15px;
                border-radius: 5px;
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>Enhanced Interactive 3D Coach Clustering Visualization</h1>
        
        <div class="controls">
            <div class="control-row">
                <div class="control-group">
                    <label for="year-min">Year Range:</label>
                    <div style="display: flex; gap: 10px; align-items: center;">
                        <input type="range" id="year-min" min="{df['Year'].min()}" max="{df['Year'].max()}" value="{df['Year'].min()}" oninput="updateYearRange()">
                        <span id="year-min-label">{df['Year'].min()}</span>
                        <span>to</span>
                        <input type="range" id="year-max" min="{df['Year'].min()}" max="{df['Year'].max()}" value="{df['Year'].max()}" oninput="updateYearRange()">
                        <span id="year-max-label">{df['Year'].max()}</span>
                    </div>
                </div>
                <div class="control-group">
                    <label for="coach-search">Search Coaches:</label>
                    <div style="position: relative;">
                        <input type="text" id="coach-search" placeholder="Type coach name..." oninput="searchCoaches()" onblur="hideSearchResults()">
                        <div id="search-results" class="search-results"></div>
                    </div>
                    <div id="selected-coaches" class="selected-coaches"></div>
                </div>
            </div>
            
            <div class="control-row">
                <div class="control-group">
                    <label>Show Clusters:</label>
                    <div class="cluster-checkboxes">
    """
    
    # Add cluster checkboxes
    for cluster in df['Cluster_Name'].unique():
        html_content += f"""
                        <label><input type="checkbox" value="{cluster}" checked onchange="updateFilters()"> {cluster}</label>
        """
    
    html_content += f"""
                    </div>
                </div>
                <div class="control-group">
                    <button onclick="resetFilters()">Reset All Filters</button>
                    <button onclick="showAll()">Show All Data</button>
                </div>
            </div>
        </div>
        
        <div id="plotly-div" style="height: 700px;"></div>
        
        <div id="info-panel" class="info-panel">
            <h3>Dataset Summary</h3>
            <p id="summary-text">Showing all {len(df)} coaches</p>
        </div>
        
        <script>
            // Embed the data
            const fullData = {df.to_json(orient='records')};
            const coachNames = {json.dumps(coach_names)};
            let selectedCoaches = [];
            
            // Initialize the plot with individual traces
            const clusterColors = {{
                'Modern Coordinators': '#1f77b4',
                'NFL Veterans': '#ff7f0e',
                'High Performers': '#2ca02c',
                'Position Coach Veterans': '#d62728',
                'Veteran Retreads': '#9467bd'
            }};
            
            // Create initial traces
            const initialTraces = [];
            const clusters = [...new Set(fullData.map(d => d.Cluster_Name))];
            
            clusters.forEach(cluster => {{
                const clusterData = fullData.filter(d => d.Cluster_Name === cluster);
                initialTraces.push({{
                    x: clusterData.map(d => d.tSNE_X_3D),
                    y: clusterData.map(d => d.tSNE_Y_3D),
                    z: clusterData.map(d => d.tSNE_Z_3D),
                    type: 'scatter3d',
                    mode: 'markers',
                    marker: {{
                        size: clusterData.map(d => d.Age_at_Hire * 0.3),
                        color: clusterColors[cluster],
                        opacity: 0.8,
                        line: {{
                            width: 0.5,
                            color: 'black'
                        }}
                    }},
                    name: cluster,
                    text: clusterData.map(d => d['Coach Name']),
                    hovertemplate: clusterData.map(d => 
                        `<b>${{d['Coach Name']}}</b> (${{d.Year}})<br>` +
                        `Cluster: ${{d.Cluster_Name}}<br>` +
                        `Age: ${{d.Age_at_Hire}}<br>` +
                        `NFL Coordinator: ${{d.NFL_Coordinator_Years}} years<br>` +
                        `NFL Position: ${{d.NFL_Position_Years}} years<br>` +
                        `College HC: ${{d.College_HC_Years}} years<br>` +
                        `Previous HC Jobs: ${{d.Previous_HC_Stints}}<extra></extra>`
                    )
                }});
            }});
            
            const layout = {{
                scene: {{
                    xaxis: {{ title: 't-SNE Dimension 1', gridcolor: 'lightgray', gridwidth: 1 }},
                    yaxis: {{ title: 't-SNE Dimension 2', gridcolor: 'lightgray', gridwidth: 1 }},
                    zaxis: {{ title: 't-SNE Dimension 3', gridcolor: 'lightgray', gridwidth: 1 }},
                    bgcolor: 'rgba(0,0,0,0)'
                }},
                title: {{
                    text: '<b>Enhanced Interactive 3D Coach Clustering</b><br><sub>Use controls below to filter data</sub>',
                    x: 0.5,
                    font: {{ size: 16 }}
                }},
                font: {{ size: 12 }},
                legend: {{
                    yanchor: 'top',
                    y: 1,
                    xanchor: 'left',
                    x: 0.01,
                    bgcolor: 'rgba(255,255,255,0.8)'
                }},
                height: 700,
                margin: {{ l: 0, r: 0, t: 80, b: 120 }}
            }};
            
            Plotly.newPlot('plotly-div', initialTraces, layout);
            
            function updateYearRange() {{
                const minYear = document.getElementById('year-min').value;
                const maxYear = document.getElementById('year-max').value;
                document.getElementById('year-min-label').textContent = minYear;
                document.getElementById('year-max-label').textContent = maxYear;
                updateFilters();
            }}
            
            function searchCoaches() {{
                const query = document.getElementById('coach-search').value.toLowerCase();
                const results = document.getElementById('search-results');
                
                if (query.length < 2) {{
                    results.style.display = 'none';
                    return;
                }}
                
                const matches = coachNames.filter(name => 
                    name.toLowerCase().includes(query) && !selectedCoaches.includes(name)
                ).slice(0, 10);
                
                results.innerHTML = matches.map(name => 
                    `<div class="search-item" onclick="selectCoach('${{name}}')">${{name}}</div>`
                ).join('');
                
                results.style.display = matches.length > 0 ? 'block' : 'none';
            }}
            
            function hideSearchResults() {{
                setTimeout(() => {{
                    document.getElementById('search-results').style.display = 'none';
                }}, 200);
            }}
            
            function selectCoach(name) {{
                if (!selectedCoaches.includes(name)) {{
                    selectedCoaches.push(name);
                    updateSelectedCoaches();
                    updateFilters();
                }}
                document.getElementById('coach-search').value = '';
                document.getElementById('search-results').style.display = 'none';
            }}
            
            function removeCoach(name) {{
                selectedCoaches = selectedCoaches.filter(c => c !== name);
                updateSelectedCoaches();
                updateFilters();
            }}
            
            function updateSelectedCoaches() {{
                const container = document.getElementById('selected-coaches');
                container.innerHTML = selectedCoaches.map(name => 
                    `<span class="coach-tag" onclick="removeCoach('${{name}}')">${{name}} ×</span>`
                ).join('');
            }}
            
            function updateFilters() {{
                const minYear = parseInt(document.getElementById('year-min').value);
                const maxYear = parseInt(document.getElementById('year-max').value);
                const selectedClusters = Array.from(document.querySelectorAll('input[type="checkbox"]:checked')).map(cb => cb.value);
                
                // Filter data
                const filteredData = fullData.filter(d => 
                    d.Year >= minYear && 
                    d.Year <= maxYear && 
                    selectedClusters.includes(d.Cluster_Name)
                );
                
                // Update plot with filtered data
                const newTraces = [];
                selectedClusters.forEach(cluster => {{
                    const clusterData = filteredData.filter(d => d.Cluster_Name === cluster);
                    
                    if (selectedCoaches.length > 0) {{
                        // Separate highlighted and normal coaches
                        const highlighted = clusterData.filter(d => selectedCoaches.includes(d['Coach Name']));
                        const normal = clusterData.filter(d => !selectedCoaches.includes(d['Coach Name']));
                        
                        if (normal.length > 0) {{
                            newTraces.push(createTrace(normal, cluster, false));
                        }}
                        if (highlighted.length > 0) {{
                            newTraces.push(createTrace(highlighted, cluster + ' (Selected)', true));
                        }}
                    }} else {{
                        if (clusterData.length > 0) {{
                            newTraces.push(createTrace(clusterData, cluster, false));
                        }}
                    }}
                }});
                
                Plotly.react('plotly-div', newTraces, layout);
                
                // Update summary
                document.getElementById('summary-text').textContent = 
                    `Showing ${{filteredData.length}} coaches from ${{minYear}} to ${{maxYear}}`;
            }}
            
            function createTrace(data, name, highlighted) {{
                const colors = {{
                    'Modern Coordinators': '#1f77b4',
                    'NFL Veterans': '#ff7f0e',
                    'High Performers': '#2ca02c',
                    'Position Coach Veterans': '#d62728',
                    'Veteran Retreads': '#9467bd'
                }};
                
                return {{
                    x: data.map(d => d.tSNE_X_3D),
                    y: data.map(d => d.tSNE_Y_3D),
                    z: data.map(d => d.tSNE_Z_3D),
                    type: 'scatter3d',
                    mode: 'markers',
                    marker: {{
                        size: data.map(d => d.Age_at_Hire * (highlighted ? 0.5 : 0.3)),
                        color: highlighted ? 'gold' : colors[name.replace(' (Selected)', '')],
                        opacity: highlighted ? 1.0 : 0.8,
                        line: {{
                            width: highlighted ? 3 : 0.5,
                            color: highlighted ? 'red' : 'black'
                        }}
                    }},
                    name: name,
                    text: data.map(d => d['Coach Name']),
                    hovertemplate: data.map(d => 
                        `<b>${{d['Coach Name']}}</b> (${{d.Year}})<br>` +
                        `Cluster: ${{d.Cluster_Name}}<br>` +
                        `Age: ${{d.Age_at_Hire}}<br>` +
                        `NFL Coordinator: ${{d.NFL_Coordinator_Years}} years<br>` +
                        `NFL Position: ${{d.NFL_Position_Years}} years<br>` +
                        `College HC: ${{d.College_HC_Years}} years<br>` +
                        `Previous HC Jobs: ${{d.Previous_HC_Stints}}<extra></extra>`
                    )
                }};
            }}
            
            function resetFilters() {{
                document.getElementById('year-min').value = {df['Year'].min()};
                document.getElementById('year-max').value = {df['Year'].max()};
                document.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = true);
                selectedCoaches = [];
                updateSelectedCoaches();
                updateFilters();
            }}
            
            function showAll() {{
                resetFilters();
            }}
        </script>
    </body>
    </html>
    """
    
    # Save the HTML file
    with open('enhanced_interactive_3d_clustering.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("Enhanced interactive HTML file created: enhanced_interactive_3d_clustering.html")
    print("This file includes:")
    print("- Year range filtering with sliders")
    print("- Coach search with autocomplete")
    print("- Cluster show/hide checkboxes") 
    print("- Highlighting of selected coaches")
    print("- No server required - just open the HTML file!")

if __name__ == '__main__':
    create_enhanced_interactive_plot()