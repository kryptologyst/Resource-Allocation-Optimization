"""
Visualization utilities for resource allocation optimization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

from utils.logging_config import setup_logging

logger = setup_logging(__name__)


class ResourceAllocationVisualizer:
    """Visualizer for resource allocation optimization results."""
    
    def __init__(self, style: str = "seaborn-v0_8") -> None:
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        self.style = style
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_allocation_heatmap(
        self,
        results: Dict[str, Any],
        projects: pd.DataFrame,
        resources: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create heatmap of resource allocations.
        
        Args:
            results: Optimization results
            projects: Projects DataFrame
            resources: Resources DataFrame
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure
        """
        if results["status"] != "optimal":
            logger.warning("Cannot create heatmap for failed optimization")
            return go.Figure()
        
        allocation = results["allocation"]
        
        # Create heatmap data
        project_names = projects["name"].tolist()
        resource_names = resources["name"].tolist()
        
        fig = go.Figure(data=go.Heatmap(
            z=allocation,
            x=resource_names,
            y=project_names,
            colorscale='Blues',
            hoverongaps=False,
            hovertemplate='Project: %{y}<br>Resource: %{x}<br>Allocation: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Resource Allocation Heatmap",
            xaxis_title="Resources",
            yaxis_title="Projects",
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Heatmap saved to {save_path}")
        
        return fig
    
    def plot_resource_utilization(
        self,
        results: Dict[str, Any],
        resources: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create bar chart of resource utilization.
        
        Args:
            results: Optimization results
            resources: Resources DataFrame
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure
        """
        resource_utilization = results.get("summary", {}).get("resource_utilization", {})
        
        resource_names = []
        utilization_rates = []
        used_amounts = []
        available_amounts = []
        
        for _, resource in resources.iterrows():
            resource_id = resource["resource_id"]
            util_data = resource_utilization.get(resource_id, {})
            
            resource_names.append(resource["name"])
            utilization_rates.append(util_data.get("utilization_rate", 0))
            used_amounts.append(util_data.get("used", 0))
            available_amounts.append(util_data.get("available", 0))
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Resource Utilization Rate", "Resource Usage Amounts"),
            vertical_spacing=0.1
        )
        
        # Utilization rate bar chart
        fig.add_trace(
            go.Bar(
                x=resource_names,
                y=utilization_rates,
                name="Utilization Rate",
                marker_color='lightblue',
                hovertemplate='%{x}<br>Utilization: %{y:.1%}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Usage amounts bar chart
        fig.add_trace(
            go.Bar(
                x=resource_names,
                y=used_amounts,
                name="Used",
                marker_color='lightcoral',
                hovertemplate='%{x}<br>Used: %{y}<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=resource_names,
                y=available_amounts,
                name="Available",
                marker_color='lightgreen',
                hovertemplate='%{x}<br>Available: %{y}<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Resource Utilization Analysis",
            height=800,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Resources", row=2, col=1)
        fig.update_yaxes(title_text="Utilization Rate", row=1, col=1)
        fig.update_yaxes(title_text="Amount", row=2, col=1)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Resource utilization plot saved to {save_path}")
        
        return fig
    
    def plot_project_allocation(
        self,
        results: Dict[str, Any],
        projects: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create pie chart of project allocations.
        
        Args:
            results: Optimization results
            projects: Projects DataFrame
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure
        """
        allocation_details = results.get("allocation_details", [])
        
        # Aggregate allocations by project
        project_allocations = {}
        for detail in allocation_details:
            project_id = detail["project_id"]
            allocation = detail["allocation"]
            if project_id not in project_allocations:
                project_allocations[project_id] = 0
            project_allocations[project_id] += allocation
        
        # Create pie chart data
        project_names = []
        allocation_values = []
        
        for _, project in projects.iterrows():
            project_id = project["project_id"]
            project_names.append(project["name"])
            allocation_values.append(project_allocations.get(project_id, 0))
        
        fig = go.Figure(data=[go.Pie(
            labels=project_names,
            values=allocation_values,
            hovertemplate='%{label}<br>Allocation: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Project Resource Allocation Distribution",
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Project allocation plot saved to {save_path}")
        
        return fig
    
    def plot_optimization_metrics(
        self,
        evaluation: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create radar chart of optimization metrics.
        
        Args:
            evaluation: Evaluation results
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure
        """
        business_kpis = evaluation.get("business_kpis", {})
        utilization_metrics = evaluation.get("utilization_metrics", {})
        project_metrics = evaluation.get("project_metrics", {})
        
        # Normalize metrics to 0-1 scale
        metrics = {
            "ROI": min(business_kpis.get("roi_percentage", 0) / 100, 1.0),
            "Cost Efficiency": min(business_kpis.get("cost_efficiency", 0) / 10, 1.0),
            "Resource Utilization": utilization_metrics.get("average_utilization", 0),
            "Project Completion": project_metrics.get("completion_rate", 0),
            "Utilization Efficiency": utilization_metrics.get("utilization_efficiency", 0),
            "Priority Completion": project_metrics.get("priority_completion_rate", 0)
        }
        
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Performance',
            line_color='blue'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            title="Optimization Performance Radar Chart",
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Optimization metrics plot saved to {save_path}")
        
        return fig
    
    def plot_cost_benefit_analysis(
        self,
        results: Dict[str, Any],
        projects: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create cost-benefit analysis scatter plot.
        
        Args:
            results: Optimization results
            projects: Projects DataFrame
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure
        """
        allocation_details = results.get("allocation_details", [])
        
        # Aggregate by project
        project_data = {}
        for detail in allocation_details:
            project_id = detail["project_id"]
            if project_id not in project_data:
                project_data[project_id] = {"cost": 0, "value": 0, "allocation": 0}
            project_data[project_id]["cost"] += detail["cost"]
            project_data[project_id]["value"] += detail["value"]
            project_data[project_id]["allocation"] += detail["allocation"]
        
        # Prepare data for plotting
        project_names = []
        costs = []
        values = []
        allocations = []
        colors = []
        
        for _, project in projects.iterrows():
            project_id = project["project_id"]
            if project_id in project_data:
                project_names.append(project["name"])
                costs.append(project_data[project_id]["cost"])
                values.append(project_data[project_id]["value"])
                allocations.append(project_data[project_id]["allocation"])
                colors.append(project["priority"])  # Use priority for color
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=costs,
            y=values,
            mode='markers',
            marker=dict(
                size=allocations,
                color=colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Priority")
            ),
            text=project_names,
            hovertemplate='%{text}<br>Cost: $%{x:,.0f}<br>Value: $%{y:,.0f}<br>Allocation: %{marker.size}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Cost-Benefit Analysis by Project",
            xaxis_title="Total Cost ($)",
            yaxis_title="Total Value ($)",
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Cost-benefit analysis plot saved to {save_path}")
        
        return fig
    
    def create_dashboard(
        self,
        results: Dict[str, Any],
        evaluation: Dict[str, Any],
        projects: pd.DataFrame,
        resources: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create comprehensive dashboard with multiple plots.
        
        Args:
            results: Optimization results
            evaluation: Evaluation results
            projects: Projects DataFrame
            resources: Resources DataFrame
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure with subplots
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Resource Utilization",
                "Project Allocation",
                "Cost-Benefit Analysis",
                "Performance Metrics"
            ),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "scatterpolar"}]]
        )
        
        # Resource utilization
        resource_utilization = results.get("summary", {}).get("resource_utilization", {})
        resource_names = resources["name"].tolist()
        utilization_rates = []
        
        for _, resource in resources.iterrows():
            resource_id = resource["resource_id"]
            util_data = resource_utilization.get(resource_id, {})
            utilization_rates.append(util_data.get("utilization_rate", 0))
        
        fig.add_trace(
            go.Bar(x=resource_names, y=utilization_rates, name="Utilization"),
            row=1, col=1
        )
        
        # Project allocation pie chart
        allocation_details = results.get("allocation_details", [])
        project_allocations = {}
        for detail in allocation_details:
            project_id = detail["project_id"]
            allocation = detail["allocation"]
            if project_id not in project_allocations:
                project_allocations[project_id] = 0
            project_allocations[project_id] += allocation
        
        project_names = []
        allocation_values = []
        for _, project in projects.iterrows():
            project_id = project["project_id"]
            project_names.append(project["name"])
            allocation_values.append(project_allocations.get(project_id, 0))
        
        fig.add_trace(
            go.Pie(labels=project_names, values=allocation_values, name="Allocation"),
            row=1, col=2
        )
        
        # Cost-benefit scatter
        project_data = {}
        for detail in allocation_details:
            project_id = detail["project_id"]
            if project_id not in project_data:
                project_data[project_id] = {"cost": 0, "value": 0}
            project_data[project_id]["cost"] += detail["cost"]
            project_data[project_id]["value"] += detail["value"]
        
        costs = []
        values = []
        for _, project in projects.iterrows():
            project_id = project["project_id"]
            if project_id in project_data:
                costs.append(project_data[project_id]["cost"])
                values.append(project_data[project_id]["value"])
        
        fig.add_trace(
            go.Scatter(x=costs, y=values, mode='markers', name="Cost-Benefit"),
            row=2, col=1
        )
        
        # Performance metrics radar
        business_kpis = evaluation.get("business_kpis", {})
        utilization_metrics = evaluation.get("utilization_metrics", {})
        project_metrics = evaluation.get("project_metrics", {})
        
        metrics = {
            "ROI": min(business_kpis.get("roi_percentage", 0) / 100, 1.0),
            "Utilization": utilization_metrics.get("average_utilization", 0),
            "Completion": project_metrics.get("completion_rate", 0),
            "Efficiency": utilization_metrics.get("utilization_efficiency", 0)
        }
        
        fig.add_trace(
            go.Scatterpolar(
                r=list(metrics.values()),
                theta=list(metrics.keys()),
                fill='toself',
                name="Performance"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Resource Allocation Optimization Dashboard",
            height=800,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Dashboard saved to {save_path}")
        
        return fig
