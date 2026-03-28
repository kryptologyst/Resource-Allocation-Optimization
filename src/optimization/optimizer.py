"""
Resource Allocation Optimization Module

This module provides comprehensive resource allocation optimization capabilities
for business operations, supporting multiple resource types and constraint scenarios.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from cvxpy import Variable, Problem, Maximize, Minimize, constraints
from ortools.linear_solver import pywraplp
from scipy.optimize import linprog
import pulp

from utils.config import Config
from utils.logging_config import setup_logging

logger = setup_logging(__name__)


class ResourceAllocationOptimizer:
    """
    Advanced resource allocation optimizer supporting multiple solvers and constraint types.
    
    This class provides a unified interface for resource allocation optimization,
    supporting linear programming, mixed-integer programming, and heuristic approaches.
    """
    
    def __init__(self, config: Config) -> None:
        """
        Initialize the resource allocation optimizer.
        
        Args:
            config: Configuration object containing optimization parameters
        """
        self.config = config
        self.solver_type = config.get("solver_type", "scipy")
        self.random_seed = config.get("random_seed", 42)
        
        # Set random seeds for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        logger.info(f"Initialized ResourceAllocationOptimizer with solver: {self.solver_type}")
    
    def optimize(
        self,
        projects: pd.DataFrame,
        resources: pd.DataFrame,
        allocation_matrix: pd.DataFrame,
        objective_type: str = "maximize_profit"
    ) -> Dict[str, Any]:
        """
        Optimize resource allocation across projects.
        
        Args:
            projects: DataFrame with project information
            resources: DataFrame with resource information
            allocation_matrix: DataFrame with project-resource efficiency matrix
            objective_type: Type of objective function ('maximize_profit', 'minimize_cost', 'maximize_utilization')
            
        Returns:
            Dictionary containing optimization results
        """
        logger.info(f"Starting optimization with objective: {objective_type}")
        
        # Validate inputs
        self._validate_inputs(projects, resources, allocation_matrix)
        
        # Choose solver based on configuration
        if self.solver_type == "scipy":
            result = self._solve_scipy(projects, resources, allocation_matrix, objective_type)
        elif self.solver_type == "cvxpy":
            result = self._solve_cvxpy(projects, resources, allocation_matrix, objective_type)
        elif self.solver_type == "ortools":
            result = self._solve_ortools(projects, resources, allocation_matrix, objective_type)
        elif self.solver_type == "pulp":
            result = self._solve_pulp(projects, resources, allocation_matrix, objective_type)
        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}")
        
        # Post-process results
        result = self._post_process_results(result, projects, resources, allocation_matrix)
        
        logger.info(f"Optimization completed. Status: {result['status']}")
        return result
    
    def _validate_inputs(
        self,
        projects: pd.DataFrame,
        resources: pd.DataFrame,
        allocation_matrix: pd.DataFrame
    ) -> None:
        """Validate input dataframes."""
        required_project_cols = ["project_id", "priority", "min_resources", "max_resources"]
        required_resource_cols = ["resource_id", "resource_type", "availability", "cost_per_unit"]
        required_matrix_cols = ["project_id", "resource_id", "efficiency"]
        
        for col in required_project_cols:
            if col not in projects.columns:
                raise ValueError(f"Missing required column '{col}' in projects DataFrame")
        
        for col in required_resource_cols:
            if col not in resources.columns:
                raise ValueError(f"Missing required column '{col}' in resources DataFrame")
        
        for col in required_matrix_cols:
            if col not in allocation_matrix.columns:
                raise ValueError(f"Missing required column '{col}' in allocation_matrix DataFrame")
    
    def _solve_scipy(
        self,
        projects: pd.DataFrame,
        resources: pd.DataFrame,
        allocation_matrix: pd.DataFrame,
        objective_type: str
    ) -> Dict[str, Any]:
        """Solve using scipy.optimize.linprog."""
        n_projects = len(projects)
        n_resources = len(resources)
        
        # Create objective function coefficients
        if objective_type == "maximize_profit":
            # Maximize profit = sum(efficiency * allocation) - sum(cost * allocation)
            c = []
            for i, project in projects.iterrows():
                for j, resource in resources.iterrows():
                    efficiency = allocation_matrix[
                        (allocation_matrix["project_id"] == project["project_id"]) &
                        (allocation_matrix["resource_id"] == resource["resource_id"])
                    ]["efficiency"].iloc[0] if len(allocation_matrix[
                        (allocation_matrix["project_id"] == project["project_id"]) &
                        (allocation_matrix["resource_id"] == resource["resource_id"])
                    ]) > 0 else 0
                    profit_per_unit = efficiency - resource["cost_per_unit"]
                    c.append(-profit_per_unit)  # Negative for maximization
        
        # Create constraint matrix
        A_ub = []
        b_ub = []
        
        # Resource availability constraints
        for j, resource in resources.iterrows():
            constraint = [0] * (n_projects * n_resources)
            for i, project in projects.iterrows():
                idx = i * n_resources + j
                constraint[idx] = 1
            A_ub.append(constraint)
            b_ub.append(resource["availability"])
        
        # Project constraints
        for i, project in projects.iterrows():
            # Minimum resources constraint
            constraint = [0] * (n_projects * n_resources)
            for j, resource in resources.iterrows():
                idx = i * n_resources + j
                constraint[idx] = -1  # Negative for >= constraint
            A_ub.append(constraint)
            b_ub.append(-project["min_resources"])
            
            # Maximum resources constraint
            constraint = [0] * (n_projects * n_resources)
            for j, resource in resources.iterrows():
                idx = i * n_resources + j
                constraint[idx] = 1
            A_ub.append(constraint)
            b_ub.append(project["max_resources"])
        
        # Bounds
        bounds = [(0, None)] * (n_projects * n_resources)
        
        # Solve
        result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        return {
            "status": "optimal" if result.success else "failed",
            "objective_value": -result.fun if result.success else None,
            "allocation": result.x if result.success else None,
            "message": result.message,
            "solver": "scipy"
        }
    
    def _solve_cvxpy(
        self,
        projects: pd.DataFrame,
        resources: pd.DataFrame,
        allocation_matrix: pd.DataFrame,
        objective_type: str
    ) -> Dict[str, Any]:
        """Solve using CVXPY."""
        n_projects = len(projects)
        n_resources = len(resources)
        
        # Create variables
        x = Variable((n_projects, n_resources), nonneg=True)
        
        # Create objective
        if objective_type == "maximize_profit":
            objective_terms = []
            for i, project in projects.iterrows():
                for j, resource in resources.iterrows():
                    efficiency = allocation_matrix[
                        (allocation_matrix["project_id"] == project["project_id"]) &
                        (allocation_matrix["resource_id"] == resource["resource_id"])
                    ]["efficiency"].iloc[0] if len(allocation_matrix[
                        (allocation_matrix["project_id"] == project["project_id"]) &
                        (allocation_matrix["resource_id"] == resource["resource_id"])
                    ]) > 0 else 0
                    profit_per_unit = efficiency - resource["cost_per_unit"]
                    objective_terms.append(profit_per_unit * x[i, j])
            
            objective = Maximize(sum(objective_terms))
        
        # Create constraints
        constraint_list = []
        
        # Resource availability constraints
        for j, resource in resources.iterrows():
            constraint_list.append(sum(x[:, j]) <= resource["availability"])
        
        # Project constraints
        for i, project in projects.iterrows():
            constraint_list.append(sum(x[i, :]) >= project["min_resources"])
            constraint_list.append(sum(x[i, :]) <= project["max_resources"])
        
        # Solve
        problem = Problem(objective, constraint_list)
        problem.solve()
        
        return {
            "status": "optimal" if problem.status == "optimal" else "failed",
            "objective_value": problem.value,
            "allocation": x.value,
            "message": problem.status,
            "solver": "cvxpy"
        }
    
    def _solve_ortools(
        self,
        projects: pd.DataFrame,
        resources: pd.DataFrame,
        allocation_matrix: pd.DataFrame,
        objective_type: str
    ) -> Dict[str, Any]:
        """Solve using OR-Tools."""
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            raise ValueError("Could not create OR-Tools solver")
        
        n_projects = len(projects)
        n_resources = len(resources)
        
        # Create variables
        x = {}
        for i in range(n_projects):
            for j in range(n_resources):
                x[i, j] = solver.NumVar(0, solver.infinity(), f'x_{i}_{j}')
        
        # Create objective
        objective_terms = []
        for i, project in projects.iterrows():
            for j, resource in resources.iterrows():
                efficiency = allocation_matrix[
                    (allocation_matrix["project_id"] == project["project_id"]) &
                    (allocation_matrix["resource_id"] == resource["resource_id"])
                ]["efficiency"].iloc[0] if len(allocation_matrix[
                    (allocation_matrix["project_id"] == project["project_id"]) &
                    (allocation_matrix["resource_id"] == resource["resource_id"])
                ]) > 0 else 0
                profit_per_unit = efficiency - resource["cost_per_unit"]
                objective_terms.append(profit_per_unit * x[i, j])
        
        if objective_type == "maximize_profit":
            solver.Maximize(sum(objective_terms))
        
        # Add constraints
        # Resource availability constraints
        for j, resource in resources.iterrows():
            solver.Add(sum(x[i, j] for i in range(n_projects)) <= resource["availability"])
        
        # Project constraints
        for i, project in projects.iterrows():
            solver.Add(sum(x[i, j] for j in range(n_resources)) >= project["min_resources"])
            solver.Add(sum(x[i, j] for j in range(n_resources)) <= project["max_resources"])
        
        # Solve
        status = solver.Solve()
        
        # Extract results
        allocation = np.zeros((n_projects, n_resources))
        if status == pywraplp.Solver.OPTIMAL:
            for i in range(n_projects):
                for j in range(n_resources):
                    allocation[i, j] = x[i, j].solution_value()
        
        return {
            "status": "optimal" if status == pywraplp.Solver.OPTIMAL else "failed",
            "objective_value": solver.Objective().Value() if status == pywraplp.Solver.OPTIMAL else None,
            "allocation": allocation,
            "message": f"OR-Tools status: {status}",
            "solver": "ortools"
        }
    
    def _solve_pulp(
        self,
        projects: pd.DataFrame,
        resources: pd.DataFrame,
        allocation_matrix: pd.DataFrame,
        objective_type: str
    ) -> Dict[str, Any]:
        """Solve using PuLP."""
        n_projects = len(projects)
        n_resources = len(resources)
        
        # Create problem
        if objective_type == "maximize_profit":
            prob = pulp.LpProblem("ResourceAllocation", pulp.LpMaximize)
        else:
            prob = pulp.LpProblem("ResourceAllocation", pulp.LpMinimize)
        
        # Create variables
        x = {}
        for i in range(n_projects):
            for j in range(n_resources):
                x[i, j] = pulp.LpVariable(f'x_{i}_{j}', lowBound=0)
        
        # Create objective
        objective_terms = []
        for i, project in projects.iterrows():
            for j, resource in resources.iterrows():
                efficiency = allocation_matrix[
                    (allocation_matrix["project_id"] == project["project_id"]) &
                    (allocation_matrix["resource_id"] == resource["resource_id"])
                ]["efficiency"].iloc[0] if len(allocation_matrix[
                    (allocation_matrix["project_id"] == project["project_id"]) &
                    (allocation_matrix["resource_id"] == resource["resource_id"])
                ]) > 0 else 0
                profit_per_unit = efficiency - resource["cost_per_unit"]
                objective_terms.append(profit_per_unit * x[i, j])
        
        prob += sum(objective_terms)
        
        # Add constraints
        # Resource availability constraints
        for j, resource in resources.iterrows():
            prob += sum(x[i, j] for i in range(n_projects)) <= resource["availability"]
        
        # Project constraints
        for i, project in projects.iterrows():
            prob += sum(x[i, j] for j in range(n_resources)) >= project["min_resources"]
            prob += sum(x[i, j] for j in range(n_resources)) <= project["max_resources"]
        
        # Solve
        prob.solve()
        
        # Extract results
        allocation = np.zeros((n_projects, n_resources))
        if prob.status == pulp.LpStatusOptimal:
            for i in range(n_projects):
                for j in range(n_resources):
                    allocation[i, j] = x[i, j].varValue
        
        return {
            "status": "optimal" if prob.status == pulp.LpStatusOptimal else "failed",
            "objective_value": pulp.value(prob.objective) if prob.status == pulp.LpStatusOptimal else None,
            "allocation": allocation,
            "message": pulp.LpStatus[prob.status],
            "solver": "pulp"
        }
    
    def _post_process_results(
        self,
        result: Dict[str, Any],
        projects: pd.DataFrame,
        resources: pd.DataFrame,
        allocation_matrix: pd.DataFrame
    ) -> Dict[str, Any]:
        """Post-process optimization results."""
        if result["status"] != "optimal":
            return result
        
        allocation = result["allocation"]
        n_projects, n_resources = len(projects), len(resources)
        
        # Reshape 1D allocation array to 2D if needed
        if allocation.ndim == 1:
            allocation = allocation.reshape(n_projects, n_resources)
        
        # Create detailed allocation report
        allocation_details = []
        for i, project in projects.iterrows():
            for j, resource in resources.iterrows():
                if allocation[i, j] > 0:
                    efficiency = allocation_matrix[
                        (allocation_matrix["project_id"] == project["project_id"]) &
                        (allocation_matrix["resource_id"] == resource["resource_id"])
                    ]["efficiency"].iloc[0] if len(allocation_matrix[
                        (allocation_matrix["project_id"] == project["project_id"]) &
                        (allocation_matrix["resource_id"] == resource["resource_id"])
                    ]) > 0 else 0
                    
                    allocation_details.append({
                        "project_id": project["project_id"],
                        "resource_id": resource["resource_id"],
                        "allocation": allocation[i, j],
                        "efficiency": efficiency,
                        "cost": allocation[i, j] * resource["cost_per_unit"],
                        "value": allocation[i, j] * efficiency
                    })
        
        # Calculate summary statistics
        total_cost = sum(detail["cost"] for detail in allocation_details)
        total_value = sum(detail["value"] for detail in allocation_details)
        total_profit = total_value - total_cost
        
        # Resource utilization
        resource_utilization = {}
        for j, resource in resources.iterrows():
            used = sum(allocation[i, j] for i in range(n_projects))
            resource_utilization[resource["resource_id"]] = {
                "used": used,
                "available": resource["availability"],
                "utilization_rate": used / resource["availability"] if resource["availability"] > 0 else 0
            }
        
        result.update({
            "allocation_details": allocation_details,
            "summary": {
                "total_cost": total_cost,
                "total_value": total_value,
                "total_profit": total_profit,
                "resource_utilization": resource_utilization
            }
        })
        
        return result
