"""
Benchmark visualization module for model performance metrics.
Loads performance data from Result folder and generates visualizations.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class BenchmarkLoader:
    """Load and process benchmark data from Result folder."""
    
    def __init__(self, result_dir: Path):
        """
        Initialize benchmark loader.
        
        Args:
            result_dir: Path to Result folder containing benchmark files
        """
        self.result_dir = result_dir
        self.benchmark_data = None
        self.per_class_data = None
        self.confusion_matrix_data = None
    
    def load_benchmark(self) -> Dict:
        """Load benchmark JSON data."""
        benchmark_file = self.result_dir / "benchmark_report.json"
        if benchmark_file.exists():
            with open(benchmark_file) as f:
                self.benchmark_data = json.load(f)
            return self.benchmark_data
        return {}
    
    def load_per_class(self) -> pd.DataFrame:
        """Load per-class performance CSV."""
        per_class_file = self.result_dir / "benchmark_report_per_class.csv"
        if per_class_file.exists():
            self.per_class_data = pd.read_csv(per_class_file)
            return self.per_class_data
        return pd.DataFrame()
    
    def load_confusion_matrix(self) -> pd.DataFrame:
        """Load confusion matrix CSV."""
        cm_file = self.result_dir / "benchmark_report_confusion_matrix.csv"
        if cm_file.exists():
            self.confusion_matrix_data = pd.read_csv(cm_file, index_col=0)
            return self.confusion_matrix_data
        return pd.DataFrame()
    
    def get_summary_metrics(self) -> Dict:
        """Extract summary metrics from benchmark data."""
        if not self.benchmark_data:
            self.load_benchmark()
        
        if not self.benchmark_data:
            return {}
        
        metrics = self.benchmark_data.get("aggregate_metrics", {})
        return {
            "accuracy": metrics.get("accuracy", 0),
            "macro_precision": metrics.get("macro_precision", 0),
            "macro_recall": metrics.get("macro_recall", 0),
            "macro_f1": metrics.get("macro_f1", 0),
            "weighted_f1": metrics.get("weighted_f1", 0),
            "classes": self.benchmark_data.get("summary", {}).get("classes", 0),
            "test_samples": self.benchmark_data.get("summary", {}).get("test_samples", 0),
            "inference_latency": self.benchmark_data.get("summary", {}).get("inference_latency", ""),
        }


class BenchmarkVisualizer:
    """Generate benchmark visualizations."""
    
    @staticmethod
    def generate_metrics_table(metrics: Dict) -> str:
        """
        Generate HTML table of summary metrics.
        
        Args:
            metrics: Dictionary of metrics
        
        Returns:
            HTML table string
        """
        html = """
        <div class="metrics-grid">
        """
        
        metric_items = [
            ("Accuracy", f"{metrics.get('accuracy', 0):.4f}"),
            ("Macro F1", f"{metrics.get('macro_f1', 0):.4f}"),
            ("Weighted F1", f"{metrics.get('weighted_f1', 0):.4f}"),
            ("Macro Precision", f"{metrics.get('macro_precision', 0):.4f}"),
            ("Macro Recall", f"{metrics.get('macro_recall', 0):.4f}"),
            ("Classes", str(metrics.get('classes', 0))),
            ("Test Samples", f"{metrics.get('test_samples', 0):,}"),
            ("Latency", metrics.get('inference_latency', 'N/A')),
        ]
        
        for label, value in metric_items:
            html += f"""
            <div class="metric-item">
                <span class="metric-label">{label}</span>
                <span class="metric-value">{value}</span>
            </div>
            """
        
        html += "</div>"
        return html
    
    @staticmethod
    def generate_per_class_table(df: pd.DataFrame, top_n: int = 15) -> str:
        """
        Generate HTML table of top per-class performance metrics.
        
        Args:
            df: Per-class performance DataFrame
            top_n: Number of top performers to show
        
        Returns:
            HTML table string
        """
        if df.empty:
            return "<p>No per-class data available</p>"
        
        # Sort by F1 score (descending)
        if "f1" in df.columns:
            top_classes = df.nlargest(top_n, "f1")
        else:
            top_classes = df.head(top_n)
        
        html = """
        <div class="per-class-table">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Class</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1 Score</th>
                        <th>Support</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for _, row in top_classes.iterrows():
            class_name = row.get("class", row.get("Class", "Unknown"))
            precision = f"{row.get('precision', 0):.4f}"
            recall = f"{row.get('recall', 0):.4f}"
            f1 = f"{row.get('f1', 0):.4f}"
            support = int(row.get('support', 0))
            
            # Truncate long class names
            if len(class_name) > 40:
                class_name = class_name[:37] + "..."
            
            html += f"""
                    <tr>
                        <td class="class-name">{class_name}</td>
                        <td>{precision}</td>
                        <td>{recall}</td>
                        <td><strong>{f1}</strong></td>
                        <td>{support}</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """
        
        return html
    
    @staticmethod
    def create_confusion_matrix_viz(cm_df: pd.DataFrame, output_path: Path) -> Optional[Path]:
        """
        Create confusion matrix heatmap visualization.
        
        Args:
            cm_df: Confusion matrix DataFrame
            output_path: Path to save the visualization
        
        Returns:
            Path to saved image or None
        """
        if cm_df.empty:
            return None
        
        try:
            # Create figure with appropriate size
            fig, ax = plt.subplots(figsize=(16, 14))
            
            # Create heatmap
            sns.heatmap(
                cm_df,
                annot=False,
                fmt="d",
                cmap="Blues",
                cbar_kws={"label": "Count"},
                ax=ax,
                square=True,
                xticklabels=False,
                yticklabels=False
            )
            
            ax.set_title("Confusion Matrix - All 39 Classes", fontsize=14, fontweight="bold")
            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("True Class")
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=100, bbox_inches="tight")
            plt.close()
            
            return output_path
        except Exception as e:
            print(f"Error creating confusion matrix visualization: {e}")
            return None
    
    @staticmethod
    def create_performance_distribution(df: pd.DataFrame, output_path: Path) -> Optional[Path]:
        """
        Create visualization of F1 score distribution.
        
        Args:
            df: Per-class performance DataFrame
            output_path: Path to save the visualization
        
        Returns:
            Path to saved image or None
        """
        if df.empty or "f1" not in df.columns:
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create histogram of F1 scores
            f1_scores = df["f1"].values
            ax.hist(f1_scores, bins=20, color="steelblue", edgecolor="black", alpha=0.7)
            
            # Add statistics lines
            mean = f1_scores.mean()
            median = np.median(f1_scores)
            
            ax.axvline(mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean:.3f}")
            ax.axvline(median, color="green", linestyle="--", linewidth=2, label=f"Median: {median:.3f}")
            
            ax.set_xlabel("F1 Score", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.set_title("Distribution of Per-Class F1 Scores", fontsize=14, fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=100, bbox_inches="tight")
            plt.close()
            
            return output_path
        except Exception as e:
            print(f"Error creating performance distribution: {e}")
            return None
    
    @staticmethod
    def create_metrics_comparison(df: pd.DataFrame, output_path: Path) -> Optional[Path]:
        """
        Create side-by-side comparison of precision, recall, and F1.
        
        Args:
            df: Per-class performance DataFrame
            output_path: Path to save the visualization
        
        Returns:
            Path to saved image or None
        """
        if df.empty:
            return None
        
        try:
            # Get top 20 classes by F1
            if "f1" in df.columns:
                top_df = df.nlargest(20, "f1")
            else:
                top_df = df.head(20)
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            x = np.arange(len(top_df))
            width = 0.25
            
            precision = top_df.get("precision", pd.Series([0]*len(top_df))).values
            recall = top_df.get("recall", pd.Series([0]*len(top_df))).values
            f1 = top_df.get("f1", pd.Series([0]*len(top_df))).values
            
            ax.bar(x - width, precision, width, label="Precision", color="steelblue")
            ax.bar(x, recall, width, label="Recall", color="orange")
            ax.bar(x + width, f1, width, label="F1 Score", color="green")
            
            ax.set_ylabel("Score", fontsize=12)
            ax.set_title("Top 20 Classes - Metrics Comparison", fontsize=14, fontweight="bold")
            ax.set_xticks(x)
            
            # Truncate class names for readability
            labels = [str(c)[:20] + "..." if len(str(c)) > 20 else str(c) for c in top_df.get("class", top_df.get("Class", [""])).values]
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_ylim([0, 1.1])
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=100, bbox_inches="tight")
            plt.close()
            
            return output_path
        except Exception as e:
            print(f"Error creating metrics comparison: {e}")
            return None


def load_benchmark_from_result_folder(result_dir: Path) -> Dict:
    """
    Load all benchmark data from Result folder.
    
    Args:
        result_dir: Path to Result folder
    
    Returns:
        Dictionary with loaded data and visualizations
    """
    loader = BenchmarkLoader(result_dir)
    
    # Load data
    benchmark = loader.load_benchmark()
    per_class = loader.load_per_class()
    confusion_matrix = loader.load_confusion_matrix()
    metrics = loader.get_summary_metrics()
    
    # Generate visualizations
    visualizer = BenchmarkVisualizer()
    
    metrics_html = visualizer.generate_metrics_table(metrics)
    per_class_html = visualizer.generate_per_class_table(per_class)
    
    return {
        "metrics": metrics,
        "metrics_html": metrics_html,
        "per_class_html": per_class_html,
        "per_class_df": per_class.to_dict("records") if not per_class.empty else [],
        "confusion_matrix": confusion_matrix.to_dict() if not confusion_matrix.empty else {},
        "benchmark_raw": benchmark
    }
