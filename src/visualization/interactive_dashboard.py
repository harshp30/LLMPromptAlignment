import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Any
from pathlib import Path
import base64
from io import BytesIO

class InteractiveDashboard:
    """Create interactive visualizations and dashboard for prompt alignment results."""
    
    def __init__(self, results_file: str, prompt_metadata_file: str = None):
        """Initialize the dashboard with results data."""
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        self.prompt_metadata = {}
        if prompt_metadata_file and Path(prompt_metadata_file).exists():
            with open(prompt_metadata_file, 'r') as f:
                self.prompt_metadata = json.load(f)
        
        self.augmenters = list(self.results.keys())
        self.criteria = list(self.results[self.augmenters[0]]["averages"].keys())
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
    
    def create_comprehensive_dashboard(self) -> str:
        """Create a comprehensive interactive dashboard."""
        
        # Create individual plots
        overview_fig = self._create_overview_plot()
        heatmap_fig = self._create_interactive_heatmap()
        radar_fig = self._create_radar_chart()
        distribution_fig = self._create_distribution_plot()
        correlation_fig = self._create_correlation_matrix()
        performance_fig = self._create_performance_trends()
        category_fig = self._create_category_analysis()
        
        # Generate HTML dashboard
        dashboard_html = self._generate_dashboard_html([
            ("Performance Overview", overview_fig),
            ("Detailed Heatmap", heatmap_fig),
            ("Capability Radar", radar_fig),
            ("Score Distributions", distribution_fig),
            ("Metric Correlations", correlation_fig),
            ("Performance Trends", performance_fig),
            ("Category Analysis", category_fig)
        ])
        
        # Save dashboard
        dashboard_path = self.results_dir / "interactive_dashboard.html"
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        return str(dashboard_path)
    
    def _create_overview_plot(self) -> go.Figure:
        """Create an overview bar chart with error bars and annotations."""
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=self.criteria,
            specs=[[{"secondary_y": True}] * 3] * 2
        )
        
        colors = px.colors.qualitative.Set3[:len(self.augmenters)]
        
        for i, criterion in enumerate(self.criteria):
            row = (i // 3) + 1
            col = (i % 3) + 1
            
            augmenter_scores = []
            augmenter_names = []
            
            for augmenter in self.augmenters:
                scores = []
                for prompt_scores in self.results[augmenter]["detailed"].values():
                    scores.append(prompt_scores[criterion])
                
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                
                augmenter_scores.append(mean_score)
                augmenter_names.append(augmenter.replace('_', ' ').title())
                
                # Add bar with error bar
                fig.add_trace(
                    go.Bar(
                        x=[augmenter.replace('_', ' ').title()],
                        y=[mean_score],
                        error_y=dict(type='data', array=[std_score]),
                        name=f"{augmenter} - {criterion}",
                        marker_color=colors[self.augmenters.index(augmenter)],
                        showlegend=(i == 0),
                        text=[f"{mean_score:.3f}"],
                        textposition="outside"
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title="Prompt Augmentation Performance Overview",
            height=800,
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
    
    def _create_interactive_heatmap(self) -> go.Figure:
        """Create an interactive heatmap with hover information."""
        
        # Prepare data for heatmap
        prompts = list(self.results[self.augmenters[0]]["detailed"].keys())
        
        # Create subplot for each augmenter
        fig = make_subplots(
            rows=1, cols=len(self.augmenters),
            subplot_titles=[aug.replace('_', ' ').title() for aug in self.augmenters],
            shared_yaxes=True
        )
        
        for col, augmenter in enumerate(self.augmenters, 1):
            # Create matrix
            matrix = []
            hover_text = []
            
            for prompt in prompts:
                row = []
                hover_row = []
                for criterion in self.criteria:
                    score = self.results[augmenter]["detailed"][prompt][criterion]
                    row.append(score)
                    
                    # Create hover text with metadata if available
                    metadata_info = ""
                    if prompt in self.prompt_metadata:
                        metadata = self.prompt_metadata[prompt]
                        metadata_info = f"<br>Category: {metadata.get('category', 'N/A')}<br>Risk: {metadata.get('risk_level', 'N/A')}"
                    
                    hover_row.append(
                        f"Prompt: {prompt[:50]}...<br>"
                        f"Criterion: {criterion}<br>"
                        f"Score: {score:.3f}<br>"
                        f"Augmenter: {augmenter}"
                        f"{metadata_info}"
                    )
                
                matrix.append(row)
                hover_text.append(hover_row)
            
            fig.add_trace(
                go.Heatmap(
                    z=matrix,
                    x=[c.replace('_', ' ').title() for c in self.criteria],
                    y=[p[:30] + "..." if len(p) > 30 else p for p in prompts],
                    colorscale="RdYlBu_r",
                    zmin=0, zmax=1,
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=hover_text,
                    showscale=(col == len(self.augmenters))
                ),
                row=1, col=col
            )
        
        fig.update_layout(
            title="Interactive Score Heatmap by Augmenter",
            height=max(600, len(prompts) * 20),
            template="plotly_white"
        )
        
        return fig
    
    def _create_radar_chart(self) -> go.Figure:
        """Create radar chart comparing augmenters across criteria."""
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3[:len(self.augmenters)]
        
        for i, augmenter in enumerate(self.augmenters):
            scores = [self.results[augmenter]["averages"][criterion] for criterion in self.criteria]
            
            fig.add_trace(go.Scatterpolar(
                r=scores + [scores[0]],  # Close the polygon
                theta=self.criteria + [self.criteria[0]],
                fill='toself',
                name=augmenter.replace('_', ' ').title(),
                line_color=colors[i],
                fillcolor=colors[i],
                opacity=0.6
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Capability Radar Chart",
            template="plotly_white",
            height=600
        )
        
        return fig
    
    def _create_distribution_plot(self) -> go.Figure:
        """Create distribution plots for each criterion."""
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[c.replace('_', ' ').title() for c in self.criteria]
        )
        
        colors = px.colors.qualitative.Set3[:len(self.augmenters)]
        
        for i, criterion in enumerate(self.criteria):
            row = (i // 3) + 1
            col = (i % 3) + 1
            
            for j, augmenter in enumerate(self.augmenters):
                scores = []
                for prompt_scores in self.results[augmenter]["detailed"].values():
                    scores.append(prompt_scores[criterion])
                
                fig.add_trace(
                    go.Violin(
                        y=scores,
                        name=augmenter.replace('_', ' ').title(),
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor=colors[j],
                        opacity=0.6,
                        x0=augmenter.replace('_', ' ').title(),
                        showlegend=(i == 0)
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title="Score Distributions by Criterion",
            height=800,
            template="plotly_white"
        )
        
        return fig
    
    def _create_correlation_matrix(self) -> go.Figure:
        """Create correlation matrix between different metrics."""
        
        # Prepare data for correlation analysis
        all_data = []
        
        for augmenter in self.augmenters:
            for prompt, scores in self.results[augmenter]["detailed"].items():
                row = scores.copy()
                row['augmenter'] = augmenter
                row['prompt'] = prompt
                all_data.append(row)
        
        df = pd.DataFrame(all_data)
        
        # Calculate correlation matrix for numeric columns only
        numeric_cols = [col for col in df.columns if col not in ['augmenter', 'prompt']]
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=[col.replace('_', ' ').title() for col in corr_matrix.columns],
            y=[col.replace('_', ' ').title() for col in corr_matrix.index],
            colorscale="RdBu",
            zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="Correlation: %{z:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Metric Correlation Matrix",
            height=600,
            template="plotly_white"
        )
        
        return fig
    
    def _create_performance_trends(self) -> go.Figure:
        """Create performance trends across different prompt categories."""
        
        if not self.prompt_metadata:
            # Create a simple trend based on prompt order
            fig = go.Figure()
            
            for augmenter in self.augmenters:
                overall_scores = []
                prompts = list(self.results[augmenter]["detailed"].keys())
                
                for prompt in prompts:
                    scores = list(self.results[augmenter]["detailed"][prompt].values())
                    overall_scores.append(np.mean(scores))
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(prompts))),
                    y=overall_scores,
                    mode='lines+markers',
                    name=augmenter.replace('_', ' ').title(),
                    hovertemplate="Prompt %{x}<br>Score: %{y:.3f}<extra></extra>"
                ))
            
            fig.update_layout(
                title="Performance Trends Across Prompts",
                xaxis_title="Prompt Index",
                yaxis_title="Average Score",
                template="plotly_white"
            )
            
            return fig
        
        # Create trends by category if metadata is available
        categories = set()
        for metadata in self.prompt_metadata.values():
            categories.add(metadata.get('category', 'unknown'))
        
        fig = go.Figure()
        
        for augmenter in self.augmenters:
            category_scores = {}
            
            for category in categories:
                scores = []
                for prompt, prompt_meta in self.prompt_metadata.items():
                    if prompt_meta.get('category') == category and prompt in self.results[augmenter]["detailed"]:
                        prompt_scores = list(self.results[augmenter]["detailed"][prompt].values())
                        scores.append(np.mean(prompt_scores))
                
                if scores:
                    category_scores[category] = np.mean(scores)
            
            fig.add_trace(go.Scatter(
                x=list(category_scores.keys()),
                y=list(category_scores.values()),
                mode='lines+markers',
                name=augmenter.replace('_', ' ').title(),
                hovertemplate="Category: %{x}<br>Score: %{y:.3f}<extra></extra>"
            ))
        
        fig.update_layout(
            title="Performance by Prompt Category",
            xaxis_title="Prompt Category",
            yaxis_title="Average Score",
            template="plotly_white"
        )
        
        return fig
    
    def _create_category_analysis(self) -> go.Figure:
        """Create category-based analysis if metadata is available."""
        
        if not self.prompt_metadata:
            # Create a placeholder plot
            fig = go.Figure()
            fig.add_annotation(
                text="Category analysis requires prompt metadata",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            fig.update_layout(
                title="Category Analysis",
                template="plotly_white"
            )
            return fig
        
        # Group by categories
        categories = {}
        for prompt, metadata in self.prompt_metadata.items():
            category = metadata.get('category', 'unknown')
            if category not in categories:
                categories[category] = []
            categories[category].append(prompt)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Average Scores by Category", "Risk Level Distribution", 
                          "Refusal Appropriateness", "Category Performance Comparison"],
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Average scores by category
        for i, augmenter in enumerate(self.augmenters):
            category_means = []
            category_names = []
            
            for category, prompts in categories.items():
                scores = []
                for prompt in prompts:
                    if prompt in self.results[augmenter]["detailed"]:
                        prompt_scores = list(self.results[augmenter]["detailed"][prompt].values())
                        scores.append(np.mean(prompt_scores))
                
                if scores:
                    category_means.append(np.mean(scores))
                    category_names.append(category)
            
            fig.add_trace(
                go.Bar(
                    x=category_names,
                    y=category_means,
                    name=augmenter.replace('_', ' ').title(),
                    showlegend=(i == 0)
                ),
                row=1, col=1
            )
        
        # Risk level distribution
        risk_levels = {}
        for metadata in self.prompt_metadata.values():
            risk = metadata.get('risk_level', 'unknown')
            risk_levels[risk] = risk_levels.get(risk, 0) + 1
        
        fig.add_trace(
            go.Pie(
                labels=list(risk_levels.keys()),
                values=list(risk_levels.values()),
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Category-Based Analysis",
            height=800,
            template="plotly_white"
        )
        
        return fig
    
    def _generate_dashboard_html(self, figures: List[tuple]) -> str:
        """Generate comprehensive HTML dashboard."""
        
        # Convert figures to HTML
        figure_htmls = []
        for title, fig in figures:
            fig_html = fig.to_html(include_plotlyjs=False, div_id=title.lower().replace(' ', '_'))
            figure_htmls.append((title, fig_html))
        
        # Generate comprehensive HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLM Prompt Alignment - Interactive Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f8f9fa;
                }}
                .header {{
                    text-align: center;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .dashboard-container {{
                    max-width: 1400px;
                    margin: 0 auto;
                }}
                .plot-container {{
                    background: white;
                    border-radius: 10px;
                    padding: 20px;
                    margin-bottom: 30px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .plot-title {{
                    font-size: 24px;
                    font-weight: bold;
                    margin-bottom: 15px;
                    color: #333;
                    border-bottom: 2px solid #667eea;
                    padding-bottom: 10px;
                }}
                .summary-stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .stat-card {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .stat-value {{
                    font-size: 32px;
                    font-weight: bold;
                    color: #667eea;
                }}
                .stat-label {{
                    font-size: 14px;
                    color: #666;
                    margin-top: 5px;
                }}
                .navigation {{
                    position: sticky;
                    top: 20px;
                    background: white;
                    padding: 15px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    z-index: 1000;
                }}
                .nav-links {{
                    display: flex;
                    justify-content: center;
                    gap: 20px;
                    flex-wrap: wrap;
                }}
                .nav-link {{
                    color: #667eea;
                    text-decoration: none;
                    padding: 8px 16px;
                    border-radius: 5px;
                    transition: background-color 0.3s;
                }}
                .nav-link:hover {{
                    background-color: #f0f0f0;
                }}
            </style>
        </head>
        <body>
            <div class="dashboard-container">
                <div class="header">
                    <h1>🤖 LLM Prompt Alignment Dashboard</h1>
                    <p>Comprehensive Analysis of Prompt Augmentation Strategies</p>
                    <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="navigation">
                    <div class="nav-links">
                        {' '.join([f'<a href="#{title.lower().replace(" ", "_")}" class="nav-link">{title}</a>' for title, _ in figure_htmls])}
                    </div>
                </div>
                
                <div class="summary-stats">
                    <div class="stat-card">
                        <div class="stat-value">{len(self.augmenters)}</div>
                        <div class="stat-label">Augmentation Strategies</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len(self.criteria)}</div>
                        <div class="stat-label">Evaluation Criteria</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len(list(self.results[self.augmenters[0]]["detailed"].keys()))}</div>
                        <div class="stat-label">Test Prompts</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{max([self.results[aug]["averages"][crit] for aug in self.augmenters for crit in self.criteria]):.3f}</div>
                        <div class="stat-label">Highest Score</div>
                    </div>
                </div>
        """
        
        # Add each figure
        for title, fig_html in figure_htmls:
            html_content += f"""
                <div class="plot-container" id="{title.lower().replace(' ', '_')}">
                    <div class="plot-title">{title}</div>
                    {fig_html}
                </div>
            """
        
        html_content += """
            </div>
            <script>
                // Add smooth scrolling for navigation links
                document.querySelectorAll('.nav-link').forEach(link => {
                    link.addEventListener('click', function(e) {
                        e.preventDefault();
                        const targetId = this.getAttribute('href').substring(1);
                        const targetElement = document.getElementById(targetId);
                        if (targetElement) {
                            targetElement.scrollIntoView({ behavior: 'smooth' });
                        }
                    });
                });
            </script>
        </body>
        </html>
        """
        
        return html_content 