"""
Evaluation metrics and analysis for resource allocation optimization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logging_config import setup_logging

logger = setup_logging(__name__)


class ResourceAllocationEvaluator:
    """Evaluator for resource allocation optimization results."""
    
    def __init__(self) -> None:
        """Initialize the evaluator."""
        pass
    
    def evaluate(
        self,
        results: Dict[str, Any],
        projects: pd.DataFrame,
        resources: pd.DataFrame,
        allocation_matrix: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Evaluate optimization results.
        
        Args:
            results: Optimization results
            projects: Projects DataFrame
            resources: Resources DataFrame
            allocation_matrix: Allocation matrix DataFrame
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Starting evaluation of optimization results")
        
        if results["status"] != "optimal":
            return {
                "status": "failed",
                "message": "Cannot evaluate failed optimization",
                "metrics": {}
            }
        
        # Calculate business KPIs
        business_kpis = self._calculate_business_kpis(results, projects, resources)
        
        # Calculate optimization metrics
        optimization_metrics = self._calculate_optimization_metrics(results)
        
        # Calculate resource utilization metrics
        utilization_metrics = self._calculate_utilization_metrics(results, resources)
        
        # Calculate project completion metrics
        project_metrics = self._calculate_project_metrics(results, projects)
        
        evaluation = {
            "status": "success",
            "business_kpis": business_kpis,
            "optimization_metrics": optimization_metrics,
            "utilization_metrics": utilization_metrics,
            "project_metrics": project_metrics,
            "overall_score": self._calculate_overall_score(
                business_kpis, optimization_metrics, utilization_metrics, project_metrics
            )
        }
        
        logger.info("Evaluation completed successfully")
        return evaluation
    
    def _calculate_business_kpis(
        self,
        results: Dict[str, Any],
        projects: pd.DataFrame,
        resources: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate business key performance indicators."""
        summary = results.get("summary", {})
        
        total_cost = summary.get("total_cost", 0)
        total_value = summary.get("total_value", 0)
        total_profit = summary.get("total_profit", 0)
        
        # ROI calculation
        roi = (total_profit / total_cost * 100) if total_cost > 0 else 0
        
        # Cost efficiency
        cost_efficiency = (total_value / total_cost) if total_cost > 0 else 0
        
        # Profit margin
        profit_margin = (total_profit / total_value * 100) if total_value > 0 else 0
        
        return {
            "total_cost": total_cost,
            "total_value": total_value,
            "total_profit": total_profit,
            "roi_percentage": roi,
            "cost_efficiency": cost_efficiency,
            "profit_margin_percentage": profit_margin
        }
    
    def _calculate_optimization_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimization-specific metrics."""
        objective_value = results.get("objective_value", 0)
        solver = results.get("solver", "unknown")
        
        # Solution quality metrics
        solution_quality = "high" if objective_value > 0 else "low"
        
        return {
            "objective_value": objective_value,
            "solver_used": solver,
            "solution_quality": solution_quality,
            "feasible": results.get("status") == "optimal"
        }
    
    def _calculate_utilization_metrics(
        self,
        results: Dict[str, Any],
        resources: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate resource utilization metrics."""
        resource_utilization = results.get("summary", {}).get("resource_utilization", {})
        
        utilization_rates = []
        for resource_id, util_data in resource_utilization.items():
            utilization_rates.append(util_data["utilization_rate"])
        
        avg_utilization = np.mean(utilization_rates) if utilization_rates else 0
        min_utilization = np.min(utilization_rates) if utilization_rates else 0
        max_utilization = np.max(utilization_rates) if utilization_rates else 0
        
        # Utilization efficiency (how well resources are balanced)
        utilization_efficiency = 1 - np.std(utilization_rates) if utilization_rates else 0
        
        return {
            "average_utilization": avg_utilization,
            "min_utilization": min_utilization,
            "max_utilization": max_utilization,
            "utilization_efficiency": utilization_efficiency,
            "resource_count": len(resources),
            "utilized_resources": len([r for r in utilization_rates if r > 0])
        }
    
    def _calculate_project_metrics(
        self,
        results: Dict[str, Any],
        projects: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate project-related metrics."""
        allocation_details = results.get("allocation_details", [])
        
        # Count projects with allocations
        allocated_projects = set(detail["project_id"] for detail in allocation_details)
        total_projects = len(projects)
        allocated_count = len(allocated_projects)
        
        # Project completion rate
        completion_rate = allocated_count / total_projects if total_projects > 0 else 0
        
        # Priority-weighted completion
        priority_weighted_score = 0
        total_priority_weight = 0
        
        for _, project in projects.iterrows():
            if project["project_id"] in allocated_projects:
                priority_weighted_score += project["priority"]
            total_priority_weight += project["priority"]
        
        priority_completion_rate = (
            priority_weighted_score / total_priority_weight
            if total_priority_weight > 0 else 0
        )
        
        return {
            "total_projects": total_projects,
            "allocated_projects": allocated_count,
            "completion_rate": completion_rate,
            "priority_completion_rate": priority_completion_rate,
            "unallocated_projects": total_projects - allocated_count
        }
    
    def _calculate_overall_score(
        self,
        business_kpis: Dict[str, Any],
        optimization_metrics: Dict[str, Any],
        utilization_metrics: Dict[str, Any],
        project_metrics: Dict[str, Any]
    ) -> float:
        """Calculate overall performance score."""
        # Weighted combination of different metrics
        weights = {
            "roi": 0.3,
            "utilization": 0.25,
            "completion": 0.25,
            "efficiency": 0.2
        }
        
        roi_score = min(business_kpis.get("roi_percentage", 0) / 100, 1.0)
        utilization_score = utilization_metrics.get("average_utilization", 0)
        completion_score = project_metrics.get("completion_rate", 0)
        efficiency_score = utilization_metrics.get("utilization_efficiency", 0)
        
        overall_score = (
            weights["roi"] * roi_score +
            weights["utilization"] * utilization_score +
            weights["completion"] * completion_score +
            weights["efficiency"] * efficiency_score
        )
        
        return overall_score
    
    def generate_report(
        self,
        evaluation: Dict[str, Any],
        format_type: str = "summary"
    ) -> str:
        """
        Generate evaluation report.
        
        Args:
            evaluation: Evaluation results
            format_type: Type of report ('summary', 'detailed')
            
        Returns:
            Formatted report string
        """
        if format_type == "summary":
            return self._generate_summary_report(evaluation)
        elif format_type == "detailed":
            return self._generate_detailed_report(evaluation)
        else:
            raise ValueError(f"Unknown format type: {format_type}")
    
    def _generate_summary_report(self, evaluation: Dict[str, Any]) -> str:
        """Generate summary report."""
        business_kpis = evaluation.get("business_kpis", {})
        utilization_metrics = evaluation.get("utilization_metrics", {})
        project_metrics = evaluation.get("project_metrics", {})
        overall_score = evaluation.get("overall_score", 0)
        
        report = f"""
Resource Allocation Optimization - Summary Report
================================================

Overall Performance Score: {overall_score:.2f}

Business KPIs:
- Total Profit: ${business_kpis.get('total_profit', 0):,.2f}
- ROI: {business_kpis.get('roi_percentage', 0):.1f}%
- Cost Efficiency: {business_kpis.get('cost_efficiency', 0):.2f}

Resource Utilization:
- Average Utilization: {utilization_metrics.get('average_utilization', 0):.1%}
- Resources Used: {utilization_metrics.get('utilized_resources', 0)}/{utilization_metrics.get('resource_count', 0)}

Project Completion:
- Projects Allocated: {project_metrics.get('allocated_projects', 0)}/{project_metrics.get('total_projects', 0)}
- Completion Rate: {project_metrics.get('completion_rate', 0):.1%}
"""
        return report
    
    def _generate_detailed_report(self, evaluation: Dict[str, Any]) -> str:
        """Generate detailed report."""
        # Start with summary
        report = self._generate_summary_report(evaluation)
        
        # Add detailed sections
        business_kpis = evaluation.get("business_kpis", {})
        optimization_metrics = evaluation.get("optimization_metrics", {})
        utilization_metrics = evaluation.get("utilization_metrics", {})
        project_metrics = evaluation.get("project_metrics", {})
        
        report += f"""

Detailed Metrics:
================

Business KPIs:
- Total Cost: ${business_kpis.get('total_cost', 0):,.2f}
- Total Value: ${business_kpis.get('total_value', 0):,.2f}
- Total Profit: ${business_kpis.get('total_profit', 0):,.2f}
- ROI: {business_kpis.get('roi_percentage', 0):.1f}%
- Cost Efficiency: {business_kpis.get('cost_efficiency', 0):.2f}
- Profit Margin: {business_kpis.get('profit_margin_percentage', 0):.1f}%

Optimization Metrics:
- Objective Value: {optimization_metrics.get('objective_value', 0):.2f}
- Solver Used: {optimization_metrics.get('solver_used', 'unknown')}
- Solution Quality: {optimization_metrics.get('solution_quality', 'unknown')}
- Feasible: {optimization_metrics.get('feasible', False)}

Resource Utilization:
- Average Utilization: {utilization_metrics.get('average_utilization', 0):.1%}
- Min Utilization: {utilization_metrics.get('min_utilization', 0):.1%}
- Max Utilization: {utilization_metrics.get('max_utilization', 0):.1%}
- Utilization Efficiency: {utilization_metrics.get('utilization_efficiency', 0):.2f}
- Resources Used: {utilization_metrics.get('utilized_resources', 0)}/{utilization_metrics.get('resource_count', 0)}

Project Metrics:
- Total Projects: {project_metrics.get('total_projects', 0)}
- Allocated Projects: {project_metrics.get('allocated_projects', 0)}
- Unallocated Projects: {project_metrics.get('unallocated_projects', 0)}
- Completion Rate: {project_metrics.get('completion_rate', 0):.1%}
- Priority Completion Rate: {project_metrics.get('priority_completion_rate', 0):.1%}
"""
        return report
