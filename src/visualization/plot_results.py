import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os
from pathlib import Path
from src.evaluation.alignment_metrics import AlignmentEvaluator

class ResultsVisualizer:
    """Visualizes evaluation results for different prompt augmentation strategies."""
    
    def __init__(self, results_file: str = "evaluation_results.json"):
        """
        Initialize the visualizer.
        
        Args:
            results_file (str): Path to the evaluation results JSON file
        """
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        # Create results directory if it doesn't exist
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
    
    def plot_average_scores(self, save_path: str = None):
        """Plot average scores for each augmenter."""
        if save_path is None:
            save_path = self.results_dir / "average_scores.png"
        
        # Prepare data
        augmenters = list(self.results.keys())
        criteria = list(self.results[augmenters[0]]["averages"].keys())
        
        data = []
        for augmenter in augmenters:
            for criterion, score in self.results[augmenter]["averages"].items():
                data.append({
                    "Augmenter": augmenter.replace("_", " ").title(),
                    "Criterion": criterion.replace("_", " ").title(),
                    "Score": score
                })
        
        df = pd.DataFrame(data)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=df, x="Criterion", y="Score", hue="Augmenter")
        
        plt.title("Average Scores by Criterion and Augmenter", pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.tight_layout()
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_prompt_scores(self, save_path: str = None):
        """Plot scores for each prompt and augmenter."""
        if save_path is None:
            save_path = self.results_dir / "prompt_scores.png"
        
        # Prepare data
        data = []
        for augmenter, augmenter_results in self.results.items():
            for prompt, scores in augmenter_results["detailed"].items():
                for criterion, score in scores.items():
                    data.append({
                        "Augmenter": augmenter.replace("_", " ").title(),
                        "Prompt": prompt[:30] + "..." if len(prompt) > 30 else prompt,
                        "Criterion": criterion.replace("_", " ").title(),
                        "Score": score
                    })
        
        df = pd.DataFrame(data)
        
        # Create plot
        plt.figure(figsize=(15, 8))
        ax = sns.boxplot(data=df, x="Criterion", y="Score", hue="Augmenter")
        
        plt.title("Score Distribution by Criterion and Augmenter", pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_heatmap(self, save_path: str = None):
        """Plot a heatmap of scores for each prompt and criterion."""
        # Prepare data
        augmenters = list(self.results.keys())
        prompts = list(self.results[augmenters[0]]["detailed"].keys())
        criteria = list(self.results[augmenters[0]]["detailed"][prompts[0]].keys())
        
        # Create a matrix for each augmenter
        for augmenter in augmenters:
            if save_path is None:
                save_path = self.results_dir / f"{augmenter}_score_heatmap.png"
            
            matrix = np.zeros((len(prompts), len(criteria)))
            for i, prompt in enumerate(prompts):
                for j, criterion in enumerate(criteria):
                    matrix[i, j] = self.results[augmenter]["detailed"][prompt][criterion]
            
            # Create heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                matrix,
                xticklabels=[c.replace("_", " ").title() for c in criteria],
                yticklabels=[p[:30] + "..." if len(p) > 30 else p for p in prompts],
                cmap="YlOrRd",
                vmin=0,
                vmax=1,
                annot=True,
                fmt='.2f'
            )
            
            plt.title(f"Score Heatmap - {augmenter.replace('_', ' ').title()}", pad=20)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_report(self, save_path: str = None):
        """Generate an HTML report with all visualizations and statistics."""
        if save_path is None:
            save_path = self.results_dir / "evaluation_report.html"
        
        # Create visualizations
        self.plot_average_scores()
        self.plot_prompt_scores()
        self.plot_heatmap()
        
        # Calculate statistics
        stats = {}
        for augmenter, augmenter_results in self.results.items():
            stats[augmenter] = {
                "mean": np.mean(list(augmenter_results["averages"].values())),
                "std": np.std(list(augmenter_results["averages"].values())),
                "min": np.min(list(augmenter_results["averages"].values())),
                "max": np.max(list(augmenter_results["averages"].values()))
            }
        
        # Generate HTML report
        html_content = f"""
        <html>
        <head>
            <title>Prompt Augmentation Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .section {{ margin-bottom: 30px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Prompt Augmentation Evaluation Report</h1>
                
                <div class="section">
                    <h2>Average Scores</h2>
                    <img src="average_scores.png" alt="Average Scores">
                </div>
                
                <div class="section">
                    <h2>Score Distribution</h2>
                    <img src="prompt_scores.png" alt="Prompt Scores">
                </div>
                
                <div class="section">
                    <h2>Score Heatmaps</h2>
                    <img src="system_message_score_heatmap.png" alt="System Message Heatmap">
                    <img src="few_shot_score_heatmap.png" alt="Few-Shot Heatmap">
                    <img src="chain_of_thought_score_heatmap.png" alt="Chain-of-Thought Heatmap">
                </div>
                
                <div class="section">
                    <h2>Statistics</h2>
                    <table>
                        <tr>
                            <th>Augmenter</th>
                            <th>Mean Score</th>
                            <th>Std Dev</th>
                            <th>Min Score</th>
                            <th>Max Score</th>
                        </tr>
                        {self._generate_stats_rows(stats)}
                    </table>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(html_content)
    
    def _generate_stats_rows(self, stats: Dict) -> str:
        """Generate HTML table rows for statistics."""
        rows = []
        for augmenter, stat in stats.items():
            rows.append(f"""
                <tr>
                    <td>{augmenter.replace('_', ' ').title()}</td>
                    <td>{stat['mean']:.3f}</td>
                    <td>{stat['std']:.3f}</td>
                    <td>{stat['min']:.3f}</td>
                    <td>{stat['max']:.3f}</td>
                </tr>
            """)
        return "\n".join(rows) 