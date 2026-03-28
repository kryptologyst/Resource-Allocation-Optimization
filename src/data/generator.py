"""
Data generation and processing utilities.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.config import Config
from utils.logging_config import setup_logging

logger = setup_logging(__name__)


class DataGenerator:
    """Generate synthetic data for resource allocation optimization."""
    
    def __init__(self, config: Config) -> None:
        """
        Initialize data generator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.random_seed = config.get("random_seed", 42)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
    
    def generate_projects(
        self,
        n_projects: int = 5,
        project_types: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic project data.
        
        Args:
            n_projects: Number of projects to generate
            project_types: List of project types
            
        Returns:
            DataFrame with project information
        """
        if project_types is None:
            project_types = ["Development", "Marketing", "Operations", "Research", "Support"]
        
        projects = []
        for i in range(n_projects):
            project_type = random.choice(project_types)
            priority = random.randint(1, 5)
            
            # Generate resource requirements based on project type and priority
            base_resources = {
                "Development": 50,
                "Marketing": 30,
                "Operations": 40,
                "Research": 60,
                "Support": 25
            }
            
            min_resources = base_resources[project_type] + random.randint(-10, 10)
            max_resources = min_resources + random.randint(20, 50)
            
            projects.append({
                "project_id": f"P{i+1:03d}",
                "name": f"{project_type} Project {i+1}",
                "project_type": project_type,
                "priority": priority,
                "min_resources": max(10, min_resources),
                "max_resources": max_resources,
                "deadline": f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                "expected_value": random.randint(1000, 10000)
            })
        
        return pd.DataFrame(projects)
    
    def generate_resources(
        self,
        n_resources: int = 4,
        resource_types: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic resource data.
        
        Args:
            n_resources: Number of resources to generate
            resource_types: List of resource types
            
        Returns:
            DataFrame with resource information
        """
        if resource_types is None:
            resource_types = ["Labor", "Budget", "Equipment", "Time"]
        
        resources = []
        for i in range(n_resources):
            resource_type = random.choice(resource_types)
            
            # Generate availability and cost based on resource type
            availability_ranges = {
                "Labor": (20, 100),
                "Budget": (5000, 50000),
                "Equipment": (5, 20),
                "Time": (100, 500)
            }
            
            cost_ranges = {
                "Labor": (10, 50),
                "Budget": (1, 1),
                "Equipment": (100, 500),
                "Time": (5, 20)
            }
            
            availability = random.randint(*availability_ranges[resource_type])
            cost_per_unit = random.randint(*cost_ranges[resource_type])
            
            resources.append({
                "resource_id": f"R{i+1:03d}",
                "name": f"{resource_type} Resource {i+1}",
                "resource_type": resource_type,
                "availability": availability,
                "cost_per_unit": cost_per_unit,
                "skill_level": random.randint(1, 5) if resource_type == "Labor" else None
            })
        
        return pd.DataFrame(resources)
    
    def generate_allocation_matrix(
        self,
        projects: pd.DataFrame,
        resources: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate efficiency matrix for project-resource combinations.
        
        Args:
            projects: Projects DataFrame
            resources: Resources DataFrame
            
        Returns:
            DataFrame with allocation efficiency matrix
        """
        allocation_data = []
        
        for _, project in projects.iterrows():
            for _, resource in resources.iterrows():
                # Generate efficiency based on project type and resource type
                base_efficiency = random.uniform(0.5, 1.5)
                
                # Adjust based on compatibility
                compatibility_matrix = {
                    ("Development", "Labor"): 1.2,
                    ("Development", "Budget"): 1.0,
                    ("Development", "Equipment"): 1.1,
                    ("Development", "Time"): 1.0,
                    ("Marketing", "Labor"): 1.1,
                    ("Marketing", "Budget"): 1.3,
                    ("Marketing", "Equipment"): 0.8,
                    ("Marketing", "Time"): 1.0,
                    ("Operations", "Labor"): 1.0,
                    ("Operations", "Budget"): 0.9,
                    ("Operations", "Equipment"): 1.2,
                    ("Operations", "Time"): 1.1,
                    ("Research", "Labor"): 1.3,
                    ("Research", "Budget"): 1.1,
                    ("Research", "Equipment"): 1.4,
                    ("Research", "Time"): 1.2,
                    ("Support", "Labor"): 1.1,
                    ("Support", "Budget"): 0.8,
                    ("Support", "Equipment"): 0.9,
                    ("Support", "Time"): 1.0,
                }
                
                compatibility = compatibility_matrix.get(
                    (project["project_type"], resource["resource_type"]), 1.0
                )
                
                efficiency = base_efficiency * compatibility
                
                allocation_data.append({
                    "project_id": project["project_id"],
                    "resource_id": resource["resource_id"],
                    "efficiency": efficiency,
                    "compatibility": compatibility
                })
        
        return pd.DataFrame(allocation_data)
    
    def generate_complete_dataset(
        self,
        n_projects: int = 5,
        n_resources: int = 4
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate complete synthetic dataset.
        
        Args:
            n_projects: Number of projects
            n_resources: Number of resources
            
        Returns:
            Tuple of (projects, resources, allocation_matrix) DataFrames
        """
        logger.info(f"Generating synthetic dataset: {n_projects} projects, {n_resources} resources")
        
        projects = self.generate_projects(n_projects)
        resources = self.generate_resources(n_resources)
        allocation_matrix = self.generate_allocation_matrix(projects, resources)
        
        logger.info("Synthetic dataset generation completed")
        return projects, resources, allocation_matrix
