"""
Streamlit demo application for Resource Allocation Optimization.

This application provides an interactive interface for exploring
resource allocation optimization scenarios.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.generator import DataGenerator
from src.optimization.optimizer import ResourceAllocationOptimizer
from src.evaluation.evaluator import ResourceAllocationEvaluator
from src.visualization.visualizer import ResourceAllocationVisualizer
from src.utils.config import Config


# Page configuration
st.set_page_config(
    page_title="Resource Allocation Optimization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Resource Allocation Optimization</h1>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <h4>⚠️ IMPORTANT DISCLAIMER</h4>
    <p><strong>This is an experimental research and educational tool.</strong> 
    It is NOT intended for automated decision-making without human review. 
    All optimization results should be validated by qualified professionals 
    before implementation in production environments.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("Configuration")

# Solver selection
solver_type = st.sidebar.selectbox(
    "Solver Type",
    ["scipy", "cvxpy", "ortools", "pulp"],
    index=0,
    help="Choose the optimization solver to use"
)

# Data parameters
st.sidebar.subheader("Data Parameters")
n_projects = st.sidebar.slider("Number of Projects", 3, 10, 5)
n_resources = st.sidebar.slider("Number of Resources", 2, 6, 4)

# Optimization parameters
st.sidebar.subheader("Optimization Parameters")
objective_type = st.sidebar.selectbox(
    "Objective Type",
    ["maximize_profit", "minimize_cost", "maximize_utilization"],
    index=0
)

random_seed = st.sidebar.number_input("Random Seed", value=42, min_value=1, max_value=1000)

# Scenario selection
st.sidebar.subheader("Scenario")
scenario = st.sidebar.selectbox(
    "Scenario Type",
    ["Default", "High Availability", "Constrained"],
    index=0
)

# Main content
if st.button("Run Optimization", type="primary"):
    
    # Create configuration
    config_dict = {
        "solver_type": solver_type,
        "random_seed": random_seed,
        "data_generation": {
            "n_projects": n_projects,
            "n_resources": n_resources
        },
        "optimization": {
            "objective_type": objective_type
        }
    }
    
    # Adjust configuration based on scenario
    if scenario == "High Availability":
        config_dict["data_generation"]["n_projects"] = min(8, n_projects + 2)
        config_dict["data_generation"]["n_resources"] = min(6, n_resources + 1)
    elif scenario == "Constrained":
        config_dict["data_generation"]["n_projects"] = max(3, n_projects - 1)
        config_dict["data_generation"]["n_resources"] = max(2, n_resources - 1)
    
    config = Config(**config_dict)
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Generate data
        status_text.text("Generating synthetic dataset...")
        progress_bar.progress(20)
        
        data_gen = DataGenerator(config)
        projects, resources, allocation_matrix = data_gen.generate_complete_dataset(
            n_projects=config.get("data_generation.n_projects", 5),
            n_resources=config.get("data_generation.n_resources", 4)
        )
        
        # Run optimization
        status_text.text("Running optimization...")
        progress_bar.progress(50)
        
        optimizer = ResourceAllocationOptimizer(config)
        results = optimizer.optimize(
            projects=projects,
            resources=resources,
            allocation_matrix=allocation_matrix,
            objective_type=objective_type
        )
        
        # Evaluate results
        status_text.text("Evaluating results...")
        progress_bar.progress(80)
        
        evaluator = ResourceAllocationEvaluator()
        evaluation = evaluator.evaluate(results, projects, resources, allocation_matrix)
        
        # Generate visualizations
        status_text.text("Generating visualizations...")
        progress_bar.progress(100)
        
        visualizer = ResourceAllocationVisualizer()
        
        # Display results
        st.success("Optimization completed successfully!")
        status_text.text("Complete!")
        
        # Results tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "📈 Allocations", "💰 Financial", "📋 Details", "📊 Dashboard"
        ])
        
        with tab1:
            st.subheader("Optimization Overview")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Status",
                    "✅ Optimal" if results["status"] == "optimal" else "❌ Failed",
                    delta=None
                )
            
            with col2:
                business_kpis = evaluation.get("business_kpis", {})
                st.metric(
                    "Total Profit",
                    f"${business_kpis.get('total_profit', 0):,.0f}",
                    delta=f"{business_kpis.get('roi_percentage', 0):.1f}% ROI"
                )
            
            with col3:
                utilization_metrics = evaluation.get("utilization_metrics", {})
                st.metric(
                    "Resource Utilization",
                    f"{utilization_metrics.get('average_utilization', 0):.1%}",
                    delta=f"{utilization_metrics.get('utilized_resources', 0)}/{utilization_metrics.get('resource_count', 0)} used"
                )
            
            with col4:
                project_metrics = evaluation.get("project_metrics", {})
                st.metric(
                    "Project Completion",
                    f"{project_metrics.get('completion_rate', 0):.1%}",
                    delta=f"{project_metrics.get('allocated_projects', 0)}/{project_metrics.get('total_projects', 0)} projects"
                )
            
            # Overall score
            overall_score = evaluation.get("overall_score", 0)
            st.subheader(f"Overall Performance Score: {overall_score:.2f}")
            st.progress(overall_score)
        
        with tab2:
            st.subheader("Resource Allocations")
            
            # Allocation heatmap
            if results["status"] == "optimal":
                heatmap_fig = visualizer.plot_allocation_heatmap(results, projects, resources)
                st.plotly_chart(heatmap_fig, use_container_width=True)
                
                # Resource utilization
                utilization_fig = visualizer.plot_resource_utilization(results, resources)
                st.plotly_chart(utilization_fig, use_container_width=True)
            else:
                st.error("Cannot display allocations for failed optimization")
        
        with tab3:
            st.subheader("Financial Analysis")
            
            if results["status"] == "optimal":
                # Cost-benefit analysis
                cost_benefit_fig = visualizer.plot_cost_benefit_analysis(results, projects)
                st.plotly_chart(cost_benefit_fig, use_container_width=True)
                
                # Financial metrics table
                business_kpis = evaluation.get("business_kpis", {})
                financial_data = {
                    "Metric": ["Total Cost", "Total Value", "Total Profit", "ROI", "Cost Efficiency", "Profit Margin"],
                    "Value": [
                        f"${business_kpis.get('total_cost', 0):,.2f}",
                        f"${business_kpis.get('total_value', 0):,.2f}",
                        f"${business_kpis.get('total_profit', 0):,.2f}",
                        f"{business_kpis.get('roi_percentage', 0):.1f}%",
                        f"{business_kpis.get('cost_efficiency', 0):.2f}",
                        f"{business_kpis.get('profit_margin_percentage', 0):.1f}%"
                    ]
                }
                financial_df = pd.DataFrame(financial_data)
                st.dataframe(financial_df, use_container_width=True)
            else:
                st.error("Cannot display financial analysis for failed optimization")
        
        with tab4:
            st.subheader("Detailed Results")
            
            # Projects table
            st.subheader("Projects")
            st.dataframe(projects, use_container_width=True)
            
            # Resources table
            st.subheader("Resources")
            st.dataframe(resources, use_container_width=True)
            
            # Allocation matrix
            st.subheader("Allocation Matrix")
            st.dataframe(allocation_matrix, use_container_width=True)
            
            # Allocation details
            if results["status"] == "optimal":
                st.subheader("Allocation Details")
                allocation_details = results.get("allocation_details", [])
                if allocation_details:
                    allocation_df = pd.DataFrame(allocation_details)
                    st.dataframe(allocation_df, use_container_width=True)
                else:
                    st.info("No allocation details available")
        
        with tab5:
            st.subheader("Comprehensive Dashboard")
            
            if results["status"] == "optimal":
                dashboard_fig = visualizer.create_dashboard(results, evaluation, projects, resources)
                st.plotly_chart(dashboard_fig, use_container_width=True)
            else:
                st.error("Cannot display dashboard for failed optimization")
        
        # Download results
        st.subheader("Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_projects = projects.to_csv(index=False)
            st.download_button(
                label="Download Projects CSV",
                data=csv_projects,
                file_name="projects.csv",
                mime="text/csv"
            )
        
        with col2:
            csv_resources = resources.to_csv(index=False)
            st.download_button(
                label="Download Resources CSV",
                data=csv_resources,
                file_name="resources.csv",
                mime="text/csv"
            )
        
        with col3:
            if results["status"] == "optimal":
                allocation_details = results.get("allocation_details", [])
                if allocation_details:
                    csv_allocations = pd.DataFrame(allocation_details).to_csv(index=False)
                    st.download_button(
                        label="Download Allocations CSV",
                        data=csv_allocations,
                        file_name="allocations.csv",
                        mime="text/csv"
                    )
    
    except Exception as e:
        st.error(f"Error during optimization: {str(e)}")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Resource Allocation Optimization Demo | 
    <a href="https://github.com/your-repo" target="_blank">GitHub</a> | 
    <a href="https://docs.your-site.com" target="_blank">Documentation</a></p>
</div>
""", unsafe_allow_html=True)
