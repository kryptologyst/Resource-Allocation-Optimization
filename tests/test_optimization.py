"""
Tests for resource allocation optimization.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.generator import DataGenerator
from src.optimization.optimizer import ResourceAllocationOptimizer
from src.evaluation.evaluator import ResourceAllocationEvaluator
from src.utils.config import Config


class TestDataGenerator:
    """Test data generation functionality."""
    
    def test_generate_projects(self):
        """Test project generation."""
        config = Config(random_seed=42)
        data_gen = DataGenerator(config)
        
        projects = data_gen.generate_projects(n_projects=3)
        
        assert len(projects) == 3
        assert "project_id" in projects.columns
        assert "priority" in projects.columns
        assert "min_resources" in projects.columns
        assert "max_resources" in projects.columns
    
    def test_generate_resources(self):
        """Test resource generation."""
        config = Config(random_seed=42)
        data_gen = DataGenerator(config)
        
        resources = data_gen.generate_resources(n_resources=3)
        
        assert len(resources) == 3
        assert "resource_id" in resources.columns
        assert "resource_type" in resources.columns
        assert "availability" in resources.columns
        assert "cost_per_unit" in resources.columns
    
    def test_generate_allocation_matrix(self):
        """Test allocation matrix generation."""
        config = Config(random_seed=42)
        data_gen = DataGenerator(config)
        
        projects = data_gen.generate_projects(n_projects=2)
        resources = data_gen.generate_resources(n_resources=2)
        allocation_matrix = data_gen.generate_allocation_matrix(projects, resources)
        
        assert len(allocation_matrix) == 4  # 2 projects * 2 resources
        assert "project_id" in allocation_matrix.columns
        assert "resource_id" in allocation_matrix.columns
        assert "efficiency" in allocation_matrix.columns


class TestResourceAllocationOptimizer:
    """Test optimization functionality."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        config = Config(solver_type="scipy", random_seed=42)
        optimizer = ResourceAllocationOptimizer(config)
        
        assert optimizer.solver_type == "scipy"
        assert optimizer.random_seed == 42
    
    def test_optimization_scipy(self):
        """Test optimization with scipy solver."""
        config = Config(solver_type="scipy", random_seed=42)
        optimizer = ResourceAllocationOptimizer(config)
        
        # Generate test data
        data_gen = DataGenerator(config)
        projects, resources, allocation_matrix = data_gen.generate_complete_dataset(
            n_projects=3, n_resources=2
        )
        
        # Run optimization
        results = optimizer.optimize(projects, resources, allocation_matrix)
        
        assert "status" in results
        assert "objective_value" in results
        assert "allocation" in results
        assert "solver" in results


class TestResourceAllocationEvaluator:
    """Test evaluation functionality."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = ResourceAllocationEvaluator()
        assert evaluator is not None
    
    def test_evaluation_with_optimal_results(self):
        """Test evaluation with optimal results."""
        evaluator = ResourceAllocationEvaluator()
        
        # Mock optimal results
        results = {
            "status": "optimal",
            "objective_value": 1000,
            "allocation": np.array([[10, 20], [15, 25]]),
            "allocation_details": [
                {"project_id": "P001", "resource_id": "R001", "allocation": 10, "cost": 100, "value": 200},
                {"project_id": "P001", "resource_id": "R002", "allocation": 20, "cost": 200, "value": 400},
                {"project_id": "P002", "resource_id": "R001", "allocation": 15, "cost": 150, "value": 300},
                {"project_id": "P002", "resource_id": "R002", "allocation": 25, "cost": 250, "value": 500}
            ],
            "summary": {
                "total_cost": 700,
                "total_value": 1400,
                "total_profit": 700,
                "resource_utilization": {
                    "R001": {"used": 25, "available": 50, "utilization_rate": 0.5},
                    "R002": {"used": 45, "available": 60, "utilization_rate": 0.75}
                }
            }
        }
        
        projects = pd.DataFrame({
            "project_id": ["P001", "P002"],
            "priority": [3, 4],
            "min_resources": [20, 30],
            "max_resources": [50, 60]
        })
        
        resources = pd.DataFrame({
            "resource_id": ["R001", "R002"],
            "resource_type": ["Labor", "Budget"],
            "availability": [50, 60],
            "cost_per_unit": [10, 10]
        })
        
        allocation_matrix = pd.DataFrame({
            "project_id": ["P001", "P001", "P002", "P002"],
            "resource_id": ["R001", "R002", "R001", "R002"],
            "efficiency": [2.0, 2.0, 2.0, 2.0]
        })
        
        evaluation = evaluator.evaluate(results, projects, resources, allocation_matrix)
        
        assert evaluation["status"] == "success"
        assert "business_kpis" in evaluation
        assert "optimization_metrics" in evaluation
        assert "utilization_metrics" in evaluation
        assert "project_metrics" in evaluation
        assert "overall_score" in evaluation


if __name__ == "__main__":
    pytest.main([__file__])
