#!/usr/bin/env python3
"""
Main script for running resource allocation optimization.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.generator import DataGenerator
from optimization.optimizer import ResourceAllocationOptimizer
from evaluation.evaluator import ResourceAllocationEvaluator
from evaluation.explainer import ResourceAllocationExplainer
from visualization.visualizer import ResourceAllocationVisualizer
from utils.config import Config
from utils.logging_config import setup_logging


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Resource Allocation Optimization")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Configuration file path")
    parser.add_argument("--output-dir", type=str, default="assets", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(__name__, log_level)
    
    # Load configuration
    config_path = args.config
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return 1
    
    config = Config(config_path)
    logger.info(f"Loaded configuration from {config_path}")
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "reports").mkdir(exist_ok=True)
    
    try:
        # Generate synthetic data
        logger.info("Generating synthetic dataset...")
        data_gen = DataGenerator(config)
        projects, resources, allocation_matrix = data_gen.generate_complete_dataset(
            n_projects=config.get("data_generation.n_projects", 5),
            n_resources=config.get("data_generation.n_resources", 4)
        )
        
        # Save generated data
        projects.to_csv(output_dir / "results" / "projects.csv", index=False)
        resources.to_csv(output_dir / "results" / "resources.csv", index=False)
        allocation_matrix.to_csv(output_dir / "results" / "allocation_matrix.csv", index=False)
        logger.info("Synthetic dataset generated and saved")
        
        # Run optimization
        logger.info("Running optimization...")
        optimizer = ResourceAllocationOptimizer(config)
        results = optimizer.optimize(
            projects=projects,
            resources=resources,
            allocation_matrix=allocation_matrix,
            objective_type=config.get("optimization.objective_type", "maximize_profit")
        )
        
        # Evaluate results
        logger.info("Evaluating results...")
        evaluator = ResourceAllocationEvaluator()
        evaluation = evaluator.evaluate(results, projects, resources, allocation_matrix)
        
        # Generate explainability analysis
        logger.info("Generating explainability analysis...")
        explainer = ResourceAllocationExplainer()
        explainability_report = explainer.generate_explainability_report(
            results, projects, resources, allocation_matrix, config
        )
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        visualizer = ResourceAllocationVisualizer()
        
        # Create individual plots
        plots_dir = output_dir / "plots"
        
        heatmap_fig = visualizer.plot_allocation_heatmap(results, projects, resources)
        heatmap_fig.write_html(plots_dir / "allocation_heatmap.html")
        
        utilization_fig = visualizer.plot_resource_utilization(results, resources)
        utilization_fig.write_html(plots_dir / "resource_utilization.html")
        
        project_fig = visualizer.plot_project_allocation(results, projects)
        project_fig.write_html(plots_dir / "project_allocation.html")
        
        metrics_fig = visualizer.plot_optimization_metrics(evaluation)
        metrics_fig.write_html(plots_dir / "optimization_metrics.html")
        
        cost_benefit_fig = visualizer.plot_cost_benefit_analysis(results, projects)
        cost_benefit_fig.write_html(plots_dir / "cost_benefit_analysis.html")
        
        # Create comprehensive dashboard
        dashboard_fig = visualizer.create_dashboard(results, evaluation, projects, resources)
        dashboard_fig.write_html(plots_dir / "dashboard.html")
        
        # Generate reports
        logger.info("Generating reports...")
        reports_dir = output_dir / "reports"
        
        summary_report = evaluator.generate_report(evaluation, "summary")
        with open(reports_dir / "summary_report.txt", "w") as f:
            f.write(summary_report)
        
        detailed_report = evaluator.generate_report(evaluation, "detailed")
        with open(reports_dir / "detailed_report.txt", "w") as f:
            f.write(detailed_report)
        
        # Save explainability report
        with open(reports_dir / "explainability_report.txt", "w") as f:
            f.write(explainability_report)
        
        # Save results
        import json
        with open(output_dir / "results" / "optimization_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        with open(output_dir / "results" / "evaluation_results.json", "w") as f:
            json.dump(evaluation, f, indent=2, default=str)
        
        logger.info("Optimization completed successfully!")
        logger.info(f"Results saved to {output_dir}")
        
        # Print summary
        print("\n" + "="*60)
        print("RESOURCE ALLOCATION OPTIMIZATION - SUMMARY")
        print("="*60)
        print(summary_report)
        print("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
