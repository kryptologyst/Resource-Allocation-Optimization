"""
Explainability and interpretability features for resource allocation optimization.

This module provides tools for understanding optimization decisions,
including shadow price analysis, sensitivity analysis, and decision logs.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from utils.logging_config import setup_logging

logger = setup_logging(__name__)


class ResourceAllocationExplainer:
    """Explainer for resource allocation optimization decisions."""
    
    def __init__(self) -> None:
        """Initialize the explainer."""
        self.shadow_prices: Optional[Dict[str, float]] = None
        self.sensitivity_analysis: Optional[Dict[str, Any]] = None
        self.decision_log: List[Dict[str, Any]] = []
    
    def analyze_shadow_prices(
        self,
        results: Dict[str, Any],
        projects: pd.DataFrame,
        resources: pd.DataFrame,
        allocation_matrix: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze shadow prices (dual values) for constraints.
        
        Shadow prices indicate the marginal value of relaxing each constraint.
        
        Args:
            results: Optimization results
            projects: Projects DataFrame
            resources: Resources DataFrame
            allocation_matrix: Allocation matrix DataFrame
            
        Returns:
            Dictionary containing shadow price analysis
        """
        logger.info("Analyzing shadow prices")
        
        if results["status"] != "optimal":
            logger.warning("Cannot analyze shadow prices for failed optimization")
            return {"status": "failed", "message": "Optimization not optimal"}
        
        # For linear programming, shadow prices are available in some solvers
        # This is a simplified implementation - in practice, you'd extract
        # dual values from the solver results
        
        shadow_analysis = {
            "resource_constraints": {},
            "project_constraints": {},
            "interpretation": {}
        }
        
        # Analyze resource constraint shadow prices
        resource_utilization = results.get("summary", {}).get("resource_utilization", {})
        
        for _, resource in resources.iterrows():
            resource_id = resource["resource_id"]
            util_data = resource_utilization.get(resource_id, {})
            utilization_rate = util_data.get("utilization_rate", 0)
            
            # Estimate shadow price based on utilization
            # Higher utilization = higher shadow price
            estimated_shadow_price = utilization_rate * resource["cost_per_unit"]
            
            shadow_analysis["resource_constraints"][resource_id] = {
                "resource_name": resource["name"],
                "utilization_rate": utilization_rate,
                "estimated_shadow_price": estimated_shadow_price,
                "interpretation": self._interpret_shadow_price(estimated_shadow_price, utilization_rate)
            }
        
        # Analyze project constraint shadow prices
        allocation_details = results.get("allocation_details", [])
        project_allocations = {}
        
        for detail in allocation_details:
            project_id = detail["project_id"]
            if project_id not in project_allocations:
                project_allocations[project_id] = 0
            project_allocations[project_id] += detail["allocation"]
        
        for _, project in projects.iterrows():
            project_id = project["project_id"]
            allocated = project_allocations.get(project_id, 0)
            min_resources = project["min_resources"]
            max_resources = project["max_resources"]
            
            # Estimate shadow price based on constraint tightness
            if allocated <= min_resources + 1:  # Near minimum constraint
                estimated_shadow_price = project["priority"] * 10
                interpretation = "Minimum constraint is tight - increasing minimum would improve objective"
            elif allocated >= max_resources - 1:  # Near maximum constraint
                estimated_shadow_price = -project["priority"] * 5
                interpretation = "Maximum constraint is tight - relaxing maximum would improve objective"
            else:
                estimated_shadow_price = 0
                interpretation = "Constraint is not tight"
            
            shadow_analysis["project_constraints"][project_id] = {
                "project_name": project["name"],
                "allocated": allocated,
                "min_resources": min_resources,
                "max_resources": max_resources,
                "estimated_shadow_price": estimated_shadow_price,
                "interpretation": interpretation
            }
        
        self.shadow_prices = shadow_analysis
        logger.info("Shadow price analysis completed")
        
        return shadow_analysis
    
    def _interpret_shadow_price(self, shadow_price: float, utilization_rate: float) -> str:
        """Interpret shadow price value."""
        if shadow_price > 0 and utilization_rate > 0.9:
            return "High value - constraint is tight, additional resource would significantly improve objective"
        elif shadow_price > 0 and utilization_rate > 0.7:
            return "Medium value - constraint is moderately tight"
        elif shadow_price > 0:
            return "Low value - constraint is not tight"
        else:
            return "Negative value - constraint is not binding"
    
    def perform_sensitivity_analysis(
        self,
        base_results: Dict[str, Any],
        projects: pd.DataFrame,
        resources: pd.DataFrame,
        allocation_matrix: pd.DataFrame,
        config: Any,
        variations: List[float] = [0.8, 0.9, 1.1, 1.2]
    ) -> Dict[str, Any]:
        """
        Perform sensitivity analysis by varying resource availability.
        
        Args:
            base_results: Base optimization results
            projects: Projects DataFrame
            resources: Resources DataFrame
            allocation_matrix: Allocation matrix DataFrame
            config: Configuration object
            variations: List of variation factors to test
            
        Returns:
            Dictionary containing sensitivity analysis results
        """
        logger.info("Performing sensitivity analysis")
        
        if base_results["status"] != "optimal":
            logger.warning("Cannot perform sensitivity analysis for failed optimization")
            return {"status": "failed", "message": "Base optimization not optimal"}
        
        from optimization.optimizer import ResourceAllocationOptimizer
        
        sensitivity_results = {
            "resource_variations": {},
            "objective_changes": {},
            "allocation_changes": {},
            "recommendations": []
        }
        
        base_objective = base_results.get("objective_value", 0)
        
        for _, resource in resources.iterrows():
            resource_id = resource["resource_id"]
            resource_name = resource["name"]
            
            sensitivity_results["resource_variations"][resource_id] = {
                "resource_name": resource_name,
                "base_availability": resource["availability"],
                "variations": {}
            }
            
            for variation in variations:
                # Create modified resources DataFrame
                modified_resources = resources.copy()
                modified_resources.loc[
                    modified_resources["resource_id"] == resource_id, "availability"
                ] = resource["availability"] * variation
                
                # Run optimization with modified resources
                optimizer = ResourceAllocationOptimizer(config)
                modified_results = optimizer.optimize(
                    projects=projects,
                    resources=modified_resources,
                    allocation_matrix=allocation_matrix,
                    objective_type=config.get("optimization.objective_type", "maximize_profit")
                )
                
                if modified_results["status"] == "optimal":
                    modified_objective = modified_results.get("objective_value", 0)
                    objective_change = modified_objective - base_objective
                    objective_change_pct = (objective_change / base_objective * 100) if base_objective != 0 else 0
                    
                    sensitivity_results["resource_variations"][resource_id]["variations"][variation] = {
                        "new_availability": resource["availability"] * variation,
                        "objective_value": modified_objective,
                        "objective_change": objective_change,
                        "objective_change_pct": objective_change_pct,
                        "feasible": True
                    }
                else:
                    sensitivity_results["resource_variations"][resource_id]["variations"][variation] = {
                        "new_availability": resource["availability"] * variation,
                        "feasible": False,
                        "message": modified_results.get("message", "Optimization failed")
                    }
        
        # Generate recommendations based on sensitivity analysis
        recommendations = self._generate_sensitivity_recommendations(sensitivity_results)
        sensitivity_results["recommendations"] = recommendations
        
        self.sensitivity_analysis = sensitivity_results
        logger.info("Sensitivity analysis completed")
        
        return sensitivity_results
    
    def _generate_sensitivity_recommendations(self, sensitivity_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on sensitivity analysis."""
        recommendations = []
        
        for resource_id, resource_data in sensitivity_results["resource_variations"].items():
            resource_name = resource_data["resource_name"]
            variations = resource_data["variations"]
            
            # Find the variation with highest positive impact
            best_variation = None
            best_improvement = 0
            
            for variation, data in variations.items():
                if data.get("feasible", False):
                    improvement = data.get("objective_change_pct", 0)
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_variation = variation
            
            if best_variation and best_improvement > 5:  # Significant improvement
                recommendations.append(
                    f"Increasing {resource_name} availability by "
                    f"{(best_variation - 1) * 100:.0f}% could improve objective by {best_improvement:.1f}%"
                )
        
        if not recommendations:
            recommendations.append("Current resource allocation appears optimal - no significant improvements found")
        
        return recommendations
    
    def create_decision_log(
        self,
        results: Dict[str, Any],
        projects: pd.DataFrame,
        resources: pd.DataFrame,
        config: Any
    ) -> Dict[str, Any]:
        """
        Create a comprehensive decision log for the optimization.
        
        Args:
            results: Optimization results
            projects: Projects DataFrame
            resources: Resources DataFrame
            config: Configuration object
            
        Returns:
            Dictionary containing decision log
        """
        logger.info("Creating decision log")
        
        decision_log = {
            "timestamp": datetime.now().isoformat(),
            "optimization_parameters": {
                "solver_type": config.get("solver_type", "unknown"),
                "objective_type": config.get("optimization.objective_type", "unknown"),
                "random_seed": config.get("random_seed", None)
            },
            "problem_size": {
                "n_projects": len(projects),
                "n_resources": len(resources),
                "n_variables": len(projects) * len(resources)
            },
            "optimization_results": {
                "status": results.get("status", "unknown"),
                "objective_value": results.get("objective_value", None),
                "solver_message": results.get("message", None)
            },
            "resource_summary": {},
            "project_summary": {},
            "allocation_decisions": [],
            "constraint_analysis": {},
            "recommendations": []
        }
        
        # Resource summary
        resource_utilization = results.get("summary", {}).get("resource_utilization", {})
        for _, resource in resources.iterrows():
            resource_id = resource["resource_id"]
            util_data = resource_utilization.get(resource_id, {})
            
            decision_log["resource_summary"][resource_id] = {
                "name": resource["name"],
                "type": resource["resource_type"],
                "availability": resource["availability"],
                "used": util_data.get("used", 0),
                "utilization_rate": util_data.get("utilization_rate", 0),
                "cost_per_unit": resource["cost_per_unit"]
            }
        
        # Project summary
        allocation_details = results.get("allocation_details", [])
        project_allocations = {}
        project_costs = {}
        project_values = {}
        
        for detail in allocation_details:
            project_id = detail["project_id"]
            if project_id not in project_allocations:
                project_allocations[project_id] = 0
                project_costs[project_id] = 0
                project_values[project_id] = 0
            
            project_allocations[project_id] += detail["allocation"]
            project_costs[project_id] += detail["cost"]
            project_values[project_id] += detail["value"]
        
        for _, project in projects.iterrows():
            project_id = project["project_id"]
            allocated = project_allocations.get(project_id, 0)
            cost = project_costs.get(project_id, 0)
            value = project_values.get(project_id, 0)
            
            decision_log["project_summary"][project_id] = {
                "name": project["name"],
                "priority": project["priority"],
                "min_resources": project["min_resources"],
                "max_resources": project["max_resources"],
                "allocated": allocated,
                "total_cost": cost,
                "total_value": value,
                "profit": value - cost,
                "allocated": allocated > 0
            }
        
        # Allocation decisions
        for detail in allocation_details:
            decision_log["allocation_decisions"].append({
                "project_id": detail["project_id"],
                "resource_id": detail["resource_id"],
                "allocation": detail["allocation"],
                "efficiency": detail["efficiency"],
                "cost": detail["cost"],
                "value": detail["value"],
                "decision_rationale": self._generate_decision_rationale(detail, projects, resources)
            })
        
        # Constraint analysis
        decision_log["constraint_analysis"] = {
            "binding_constraints": [],
            "slack_constraints": [],
            "recommendations": []
        }
        
        # Identify binding constraints
        for _, resource in resources.iterrows():
            resource_id = resource["resource_id"]
            util_data = resource_utilization.get(resource_id, {})
            utilization_rate = util_data.get("utilization_rate", 0)
            
            if utilization_rate > 0.95:
                decision_log["constraint_analysis"]["binding_constraints"].append(
                    f"Resource {resource['name']} is fully utilized ({utilization_rate:.1%})"
                )
            elif utilization_rate < 0.1:
                decision_log["constraint_analysis"]["slack_constraints"].append(
                    f"Resource {resource['name']} is underutilized ({utilization_rate:.1%})"
                )
        
        # Generate recommendations
        recommendations = []
        
        # Resource recommendations
        for resource_id, util_data in resource_utilization.items():
            utilization_rate = util_data.get("utilization_rate", 0)
            if utilization_rate > 0.9:
                recommendations.append(f"Consider increasing availability of resource {resource_id}")
            elif utilization_rate < 0.2:
                recommendations.append(f"Consider reducing availability of resource {resource_id}")
        
        # Project recommendations
        for project_id, project_data in decision_log["project_summary"].items():
            if not project_data["allocated"] and project_data["priority"] >= 4:
                recommendations.append(f"High-priority project {project_id} was not allocated resources")
        
        decision_log["recommendations"] = recommendations
        
        # Store in decision log
        self.decision_log.append(decision_log)
        
        logger.info("Decision log created")
        return decision_log
    
    def _generate_decision_rationale(self, detail: Dict[str, Any], projects: pd.DataFrame, resources: pd.DataFrame) -> str:
        """Generate rationale for allocation decision."""
        project_id = detail["project_id"]
        resource_id = detail["resource_id"]
        allocation = detail["allocation"]
        efficiency = detail["efficiency"]
        
        project = projects[projects["project_id"] == project_id].iloc[0]
        resource = resources[resources["resource_id"] == resource_id].iloc[0]
        
        rationale = f"Allocated {allocation:.1f} units of {resource['name']} to {project['name']} "
        rationale += f"(efficiency: {efficiency:.2f}, priority: {project['priority']})"
        
        return rationale
    
    def generate_explainability_report(
        self,
        results: Dict[str, Any],
        projects: pd.DataFrame,
        resources: pd.DataFrame,
        allocation_matrix: pd.DataFrame,
        config: Any
    ) -> str:
        """
        Generate comprehensive explainability report.
        
        Args:
            results: Optimization results
            projects: Projects DataFrame
            resources: Resources DataFrame
            allocation_matrix: Allocation matrix DataFrame
            config: Configuration object
            
        Returns:
            Formatted explainability report
        """
        logger.info("Generating explainability report")
        
        # Perform all analyses
        shadow_analysis = self.analyze_shadow_prices(results, projects, resources, allocation_matrix)
        sensitivity_analysis = self.perform_sensitivity_analysis(results, projects, resources, allocation_matrix, config)
        decision_log = self.create_decision_log(results, projects, resources, config)
        
        report = f"""
Resource Allocation Optimization - Explainability Report
======================================================

Generated: {decision_log['timestamp']}

1. OPTIMIZATION SUMMARY
-----------------------
Solver: {decision_log['optimization_parameters']['solver_type']}
Objective: {decision_log['optimization_parameters']['objective_type']}
Status: {decision_log['optimization_results']['status']}
Objective Value: {decision_log['optimization_results']['objective_value']:.2f}

Problem Size:
- Projects: {decision_log['problem_size']['n_projects']}
- Resources: {decision_log['problem_size']['n_resources']}
- Variables: {decision_log['problem_size']['n_variables']}

2. SHADOW PRICE ANALYSIS
-------------------------
"""
        
        if shadow_analysis.get("status") == "success":
            report += "\nResource Constraints:\n"
            for resource_id, data in shadow_analysis["resource_constraints"].items():
                report += f"- {data['resource_name']}: {data['estimated_shadow_price']:.2f} "
                report += f"(utilization: {data['utilization_rate']:.1%})\n"
                report += f"  {data['interpretation']}\n"
            
            report += "\nProject Constraints:\n"
            for project_id, data in shadow_analysis["project_constraints"].items():
                report += f"- {data['project_name']}: {data['estimated_shadow_price']:.2f}\n"
                report += f"  {data['interpretation']}\n"
        else:
            report += "Shadow price analysis not available for failed optimization.\n"
        
        report += f"""
3. SENSITIVITY ANALYSIS
-----------------------
"""
        
        if sensitivity_analysis.get("status") == "success":
            report += "\nResource Sensitivity:\n"
            for resource_id, data in sensitivity_analysis["resource_variations"].items():
                resource_name = data["resource_name"]
                report += f"- {resource_name}:\n"
                for variation, var_data in data["variations"].items():
                    if var_data.get("feasible", False):
                        change_pct = var_data.get("objective_change_pct", 0)
                        report += f"  {variation:.1f}x availability: {change_pct:+.1f}% objective change\n"
            
            report += "\nRecommendations:\n"
            for rec in sensitivity_analysis.get("recommendations", []):
                report += f"- {rec}\n"
        else:
            report += "Sensitivity analysis not available for failed optimization.\n"
        
        report += f"""
4. DECISION LOG SUMMARY
-----------------------
"""
        
        # Resource utilization summary
        report += "\nResource Utilization:\n"
        for resource_id, data in decision_log["resource_summary"].items():
            report += f"- {data['name']}: {data['used']:.1f}/{data['availability']:.1f} "
            report += f"({data['utilization_rate']:.1%})\n"
        
        # Project allocation summary
        report += "\nProject Allocations:\n"
        for project_id, data in decision_log["project_summary"].items():
            status = "Allocated" if data["allocated"] else "Not Allocated"
            report += f"- {data['name']}: {status} "
            if data["allocated"]:
                report += f"(Priority: {data['priority']}, Profit: ${data['profit']:.2f})\n"
            else:
                report += f"(Priority: {data['priority']})\n"
        
        # Constraint analysis
        report += "\nConstraint Analysis:\n"
        binding_constraints = decision_log["constraint_analysis"]["binding_constraints"]
        slack_constraints = decision_log["constraint_analysis"]["slack_constraints"]
        
        if binding_constraints:
            report += "Binding Constraints:\n"
            for constraint in binding_constraints:
                report += f"- {constraint}\n"
        
        if slack_constraints:
            report += "Slack Constraints:\n"
            for constraint in slack_constraints:
                report += f"- {constraint}\n"
        
        # Recommendations
        report += "\nRecommendations:\n"
        for rec in decision_log["recommendations"]:
            report += f"- {rec}\n"
        
        report += f"""
5. ALLOCATION DECISIONS
-----------------------
"""
        
        for decision in decision_log["allocation_decisions"]:
            report += f"- {decision['decision_rationale']}\n"
        
        report += f"""
6. INTERPRETATION GUIDANCE
--------------------------
- Shadow prices indicate the marginal value of relaxing constraints
- Positive shadow prices suggest tight constraints that limit the objective
- Sensitivity analysis shows how objective changes with resource variations
- Decision log provides traceability for all allocation choices
- Recommendations help identify potential improvements

Note: This analysis is for educational and research purposes only.
All decisions should be validated by qualified professionals.
"""
        
        logger.info("Explainability report generated")
        return report
