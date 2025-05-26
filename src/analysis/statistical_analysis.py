import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.metrics import cohen_kappa_score
from typing import Dict, List, Tuple, Any
import json
from pathlib import Path

class AdvancedStatisticalAnalyzer:
    """Advanced statistical analysis for prompt augmentation evaluation results."""
    
    def __init__(self, results_file: str):
        """Initialize with evaluation results."""
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        self.augmenters = list(self.results.keys())
        self.criteria = list(self.results[self.augmenters[0]]["averages"].keys())
        
    def perform_significance_testing(self) -> Dict[str, Any]:
        """Perform statistical significance tests between augmenters."""
        results = {}
        
        # Prepare data for each criterion
        for criterion in self.criteria:
            criterion_data = {}
            for augmenter in self.augmenters:
                scores = []
                for prompt_scores in self.results[augmenter]["detailed"].values():
                    scores.append(prompt_scores[criterion])
                criterion_data[augmenter] = scores
            
            # Friedman test (non-parametric ANOVA for related samples)
            augmenter_scores = [criterion_data[aug] for aug in self.augmenters]
            friedman_stat, friedman_p = friedmanchisquare(*augmenter_scores)
            
            # Pairwise Wilcoxon signed-rank tests
            pairwise_tests = {}
            for i, aug1 in enumerate(self.augmenters):
                for j, aug2 in enumerate(self.augmenters[i+1:], i+1):
                    stat, p_val = wilcoxon(criterion_data[aug1], criterion_data[aug2])
                    pairwise_tests[f"{aug1}_vs_{aug2}"] = {
                        "statistic": stat,
                        "p_value": p_val,
                        "significant": p_val < 0.05
                    }
            
            results[criterion] = {
                "friedman_test": {
                    "statistic": friedman_stat,
                    "p_value": friedman_p,
                    "significant": friedman_p < 0.05
                },
                "pairwise_tests": pairwise_tests
            }
        
        return results
    
    def calculate_effect_sizes(self) -> Dict[str, Dict[str, float]]:
        """Calculate Cohen's d effect sizes between augmenters."""
        effect_sizes = {}
        
        for criterion in self.criteria:
            criterion_effects = {}
            
            # Get scores for each augmenter
            augmenter_scores = {}
            for augmenter in self.augmenters:
                scores = []
                for prompt_scores in self.results[augmenter]["detailed"].values():
                    scores.append(prompt_scores[criterion])
                augmenter_scores[augmenter] = np.array(scores)
            
            # Calculate pairwise effect sizes
            for i, aug1 in enumerate(self.augmenters):
                for j, aug2 in enumerate(self.augmenters[i+1:], i+1):
                    scores1 = augmenter_scores[aug1]
                    scores2 = augmenter_scores[aug2]
                    
                    # Cohen's d
                    pooled_std = np.sqrt(((len(scores1) - 1) * np.var(scores1, ddof=1) + 
                                        (len(scores2) - 1) * np.var(scores2, ddof=1)) / 
                                       (len(scores1) + len(scores2) - 2))
                    
                    cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
                    criterion_effects[f"{aug1}_vs_{aug2}"] = cohens_d
            
            effect_sizes[criterion] = criterion_effects
        
        return effect_sizes
    
    def calculate_reliability_metrics(self) -> Dict[str, float]:
        """Calculate inter-rater reliability and consistency metrics."""
        # Simulate multiple evaluations for reliability analysis
        # In practice, you'd have multiple human raters or evaluation runs
        
        reliability_metrics = {}
        
        for criterion in self.criteria:
            scores_matrix = []
            for augmenter in self.augmenters:
                augmenter_scores = []
                for prompt_scores in self.results[augmenter]["detailed"].values():
                    augmenter_scores.append(prompt_scores[criterion])
                scores_matrix.append(augmenter_scores)
            
            # Calculate Cronbach's alpha (internal consistency)
            scores_df = pd.DataFrame(scores_matrix).T
            cronbach_alpha = self._cronbach_alpha(scores_df)
            reliability_metrics[f"{criterion}_cronbach_alpha"] = cronbach_alpha
            
            # Calculate ICC (Intraclass Correlation Coefficient)
            icc = self._calculate_icc(scores_df)
            reliability_metrics[f"{criterion}_icc"] = icc
        
        return reliability_metrics
    
    def _cronbach_alpha(self, df: pd.DataFrame) -> float:
        """Calculate Cronbach's alpha for internal consistency."""
        df_corr = df.corr()
        N = df.shape[1]
        rs = np.array([df_corr.values[i, j] for i in range(N) for j in range(i+1, N)])
        mean_r = np.mean(rs)
        cronbach_alpha = (N * mean_r) / (1 + (N - 1) * mean_r)
        return cronbach_alpha
    
    def _calculate_icc(self, df: pd.DataFrame) -> float:
        """Calculate Intraclass Correlation Coefficient."""
        # Simplified ICC calculation
        n_subjects, n_raters = df.shape
        
        # Calculate means
        subject_means = df.mean(axis=1)
        rater_means = df.mean(axis=0)
        grand_mean = df.values.mean()
        
        # Calculate sum of squares
        ss_total = np.sum((df.values - grand_mean) ** 2)
        ss_between_subjects = n_raters * np.sum((subject_means - grand_mean) ** 2)
        ss_between_raters = n_subjects * np.sum((rater_means - grand_mean) ** 2)
        ss_error = ss_total - ss_between_subjects - ss_between_raters
        
        # Calculate mean squares
        ms_between_subjects = ss_between_subjects / (n_subjects - 1)
        ms_error = ss_error / ((n_subjects - 1) * (n_raters - 1))
        
        # ICC(2,1) - two-way random effects, single measures
        icc = (ms_between_subjects - ms_error) / (ms_between_subjects + (n_raters - 1) * ms_error)
        return max(0, icc)  # ICC should be non-negative
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive statistical analysis report."""
        report = {
            "significance_tests": self.perform_significance_testing(),
            "effect_sizes": self.calculate_effect_sizes(),
            "reliability_metrics": self.calculate_reliability_metrics(),
            "descriptive_statistics": self._calculate_descriptive_stats(),
            "performance_rankings": self._calculate_performance_rankings()
        }
        
        return report
    
    def _calculate_descriptive_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculate comprehensive descriptive statistics."""
        stats_dict = {}
        
        for criterion in self.criteria:
            criterion_stats = {}
            for augmenter in self.augmenters:
                scores = []
                for prompt_scores in self.results[augmenter]["detailed"].values():
                    scores.append(prompt_scores[criterion])
                
                scores = np.array(scores)
                criterion_stats[augmenter] = {
                    "mean": np.mean(scores),
                    "median": np.median(scores),
                    "std": np.std(scores, ddof=1),
                    "min": np.min(scores),
                    "max": np.max(scores),
                    "q25": np.percentile(scores, 25),
                    "q75": np.percentile(scores, 75),
                    "skewness": stats.skew(scores),
                    "kurtosis": stats.kurtosis(scores),
                    "cv": np.std(scores, ddof=1) / np.mean(scores) if np.mean(scores) != 0 else 0
                }
            
            stats_dict[criterion] = criterion_stats
        
        return stats_dict
    
    def _calculate_performance_rankings(self) -> Dict[str, List[str]]:
        """Calculate performance rankings for each criterion."""
        rankings = {}
        
        for criterion in self.criteria:
            augmenter_means = []
            for augmenter in self.augmenters:
                mean_score = self.results[augmenter]["averages"][criterion]
                augmenter_means.append((augmenter, mean_score))
            
            # Sort by score (descending)
            augmenter_means.sort(key=lambda x: x[1], reverse=True)
            rankings[criterion] = [aug[0] for aug in augmenter_means]
        
        return rankings 