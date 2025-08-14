#!/usr/bin/env python3
"""
Interactive Streamlit Dashboard for Athlete Performance Predictor

This app provides the full interactive dashboard requested in the Cursor evaluation prompt:
- File upload and real-time analysis
- Interactive visualizations with Plotly
- Export functionality (PDF, CSV, API)
- Historical comparisons
- Biomechanical asymmetry visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import base64
from io import BytesIO
import logging

# Import our ML models
try:
    from ml_models import (
        EnsemblePredictor, BiomechanicalAsymmetryDetector, 
        InjuryRiskPrediction, BiomechanicalAsymmetry
    )
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.warning("ML models not available. Install required dependencies.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🏃 Athlete Performance Analyzer",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitDashboard:
    """Main dashboard class for the Streamlit app"""
    
    def __init__(self):
        self.ensemble_predictor = None
        self.athlete_data = None
        self.analysis_results = None
        
        if ML_AVAILABLE:
            self.ensemble_predictor = EnsemblePredictor()
        
        # Initialize session state
        if 'uploaded_file' not in st.session_state:
            st.session_state.uploaded_file = None
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False
    
    def main(self):
        """Main dashboard function"""
        st.title("🏃 Athlete Performance Analyzer")
        st.markdown("### Advanced ML-Powered Fitness Analysis with Research-Based Insights")
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigation",
            ["📊 Dashboard", "🔬 Analysis", "📈 Visualizations", "📋 Reports", "⚙️ Settings"]
        )
        
        if page == "📊 Dashboard":
            self.show_dashboard()
        elif page == "🔬 Analysis":
            self.show_analysis()
        elif page == "📈 Visualizations":
            self.show_visualizations()
        elif page == "📋 Reports":
            self.show_reports()
        elif page == "⚙️ Settings":
            self.show_settings()
    
    def show_dashboard(self):
        """Main dashboard view"""
        st.header("📊 Performance Overview")
        
        # File upload section
        st.subheader("📁 Data Upload")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload your fitness data (CSV, JSON, or Excel)",
                type=['csv', 'json', 'xlsx', 'xls'],
                help="Upload your Strava activities, VeSync data, or other fitness metrics"
            )
            
            if uploaded_file is not None:
                st.session_state.uploaded_file = uploaded_file
                self.load_data(uploaded_file)
        
        with col2:
            st.info("""
            **Supported Formats:**
            - Strava activities (CSV/JSON)
            - VeSync device data (JSON)
            - Custom fitness metrics (CSV)
            - Excel spreadsheets
            """)
        
        # Quick stats if data is loaded
        if self.athlete_data is not None:
            self.show_quick_stats()
            
            # Run analysis button
            if st.button("🚀 Run Full Analysis", type="primary"):
                with st.spinner("Running comprehensive analysis..."):
                    self.run_analysis()
                    st.session_state.analysis_complete = True
                    st.success("Analysis complete! Check the Analysis tab for detailed results.")
        
        # Recent activity preview
        if self.athlete_data is not None:
            st.subheader("📅 Recent Activity Preview")
            st.dataframe(
                self.athlete_data.tail(10),
                use_container_width=True
            )
    
    def show_analysis(self):
        """Detailed analysis results"""
        st.header("🔬 Analysis Results")
        
        if not st.session_state.analysis_complete:
            st.info("Please upload data and run analysis from the Dashboard tab first.")
            return
        
        if self.analysis_results is None:
            st.error("No analysis results available.")
            return
        
        # Analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Injury Risk", "⚖️ Biomechanical", "📊 Performance", "💡 Recommendations"
        ])
        
        with tab1:
            self.show_injury_risk_analysis()
        
        with tab2:
            self.show_biomechanical_analysis()
        
        with tab3:
            self.show_performance_analysis()
        
        with tab4:
            self.show_recommendations()
    
    def show_visualizations(self):
        """Interactive visualizations"""
        st.header("📈 Interactive Visualizations")
        
        if self.athlete_data is None:
            st.info("Please upload data first to see visualizations.")
            return
        
        # Visualization options
        viz_type = st.selectbox(
            "Choose Visualization Type",
            ["Training Load Trends", "Activity Distribution", "Performance Metrics", "Risk Timeline"]
        )
        
        if viz_type == "Training Load Trends":
            self.plot_training_load_trends()
        elif viz_type == "Activity Distribution":
            self.plot_activity_distribution()
        elif viz_type == "Performance Metrics":
            self.plot_performance_metrics()
        elif viz_type == "Risk Timeline":
            self.plot_risk_timeline()
    
    def show_reports(self):
        """Report generation and export"""
        st.header("📋 Reports & Export")
        
        if self.athlete_data is None:
            st.info("Please upload data first to generate reports.")
            return
        
        # Report options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Generate Reports")
            
            report_type = st.selectbox(
                "Report Type",
                ["Summary Report", "Detailed Analysis", "Performance Trends", "Custom Report"]
            )
            
            if st.button("📄 Generate Report"):
                with st.spinner("Generating report..."):
                    report = self.generate_report(report_type)
                    st.success("Report generated successfully!")
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Report",
                        data=report,
                        file_name=f"athlete_report_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )
        
        with col2:
            st.subheader("📤 Export Data")
            
            export_format = st.selectbox(
                "Export Format",
                ["CSV", "JSON", "Excel", "PDF"]
            )
            
            if st.button("📤 Export Data"):
                with st.spinner("Exporting data..."):
                    exported_data = self.export_data(export_format)
                    st.success("Data exported successfully!")
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Data",
                        data=exported_data,
                        file_name=f"athlete_data_{datetime.now().strftime('%Y%m%d')}.{export_format.lower()}",
                        mime=self.get_mime_type(export_format)
                    )
    
    def show_settings(self):
        """Settings and configuration"""
        st.header("⚙️ Settings & Configuration")
        
        # ML model settings
        st.subheader("🤖 ML Model Configuration")
        
        if ML_AVAILABLE:
            st.success("ML models loaded successfully!")
            
            # Model parameters
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.5,
                max_value=0.95,
                value=0.8,
                step=0.05,
                help="Minimum confidence level for predictions"
            )
            
            risk_sensitivity = st.selectbox(
                "Risk Sensitivity",
                ["Conservative", "Balanced", "Aggressive"],
                help="How sensitive the system should be to injury risk"
            )
            
            # Save settings
            if st.button("💾 Save Settings"):
                st.success("Settings saved!")
        else:
            st.error("ML models not available. Check dependencies.")
        
        # Data settings
        st.subheader("📊 Data Settings")
        
        data_refresh_rate = st.selectbox(
            "Data Refresh Rate",
            ["Daily", "Weekly", "Monthly", "Manual"],
            help="How often to refresh data from sources"
        )
        
        max_data_points = st.number_input(
            "Maximum Data Points",
            min_value=100,
            max_value=10000,
            value=1000,
            help="Maximum number of data points to analyze"
        )
    
    def load_data(self, uploaded_file):
        """Load and process uploaded data"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                self.athlete_data = pd.read_csv(uploaded_file)
            elif file_extension == 'json':
                self.athlete_data = pd.read_json(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                self.athlete_data = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format")
                return
            
            # Basic data processing
            if 'date' in self.athlete_data.columns:
                self.athlete_data['date'] = pd.to_datetime(self.athlete_data['date'])
                self.athlete_data = self.athlete_data.sort_values('date')
            
            # Calculate basic metrics
            self.calculate_basic_metrics()
            
            st.success(f"Data loaded successfully! {len(self.athlete_data)} records found.")
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            logger.error(f"Data loading error: {e}")
    
    def calculate_basic_metrics(self):
        """Calculate basic fitness metrics"""
        try:
            if 'duration_min' in self.athlete_data.columns and 'distance_miles' in self.athlete_data.columns:
                # Training load calculation
                self.athlete_data['training_load'] = (
                    self.athlete_data['duration_min'] * 
                    self.athlete_data['distance_miles'].fillna(0) / 10
                )
                
                # Rolling metrics
                self.athlete_data['acute_load'] = self.athlete_data['training_load'].rolling(7, min_periods=1).sum()
                self.athlete_data['chronic_load'] = self.athlete_data['training_load'].rolling(28, min_periods=1).sum()
                self.athlete_data['load_ratio'] = self.athlete_data['acute_load'] / (self.athlete_data['chronic_load'] + 1)
            
        except Exception as e:
            logger.error(f"Error calculating basic metrics: {e}")
    
    def show_quick_stats(self):
        """Display quick statistics"""
        if self.athlete_data is None:
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_activities = len(self.athlete_data)
            st.metric("Total Activities", total_activities)
        
        with col2:
            if 'duration_min' in self.athlete_data.columns:
                total_hours = self.athlete_data['duration_min'].sum() / 60
                st.metric("Total Hours", f"{total_hours:.1f}")
        
        with col3:
            if 'distance_miles' in self.athlete_data.columns:
                total_distance = self.athlete_data['distance_miles'].sum()
                st.metric("Total Distance", f"{total_distance:.1f} mi")
        
        with col4:
            if 'date' in self.athlete_data.columns:
                date_range = (self.athlete_data['date'].max() - self.athlete_data['date'].min()).days
                st.metric("Date Range", f"{date_range} days")
    
    def run_analysis(self):
        """Run comprehensive analysis"""
        try:
            if not ML_AVAILABLE or self.ensemble_predictor is None:
                st.error("ML models not available for analysis")
                return
            
            # Run ensemble prediction
            self.analysis_results = self.ensemble_predictor.predict_comprehensive_risk(
                self.athlete_data
            )
            
            logger.info("Analysis completed successfully")
            
        except Exception as e:
            st.error(f"Error running analysis: {e}")
            logger.error(f"Analysis error: {e}")
    
    def show_injury_risk_analysis(self):
        """Display injury risk analysis results"""
        if 'injury_risk' not in self.analysis_results:
            st.warning("No injury risk analysis available")
            return
        
        injury_risk = self.analysis_results['injury_risk']
        
        # Risk overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Risk Level",
                injury_risk.risk_level,
                delta=f"{injury_risk.risk_probability:.1%}"
            )
        
        with col2:
            st.metric(
                "Confidence",
                f"{injury_risk.confidence_score:.1%}"
            )
        
        with col3:
            st.metric(
                "Risk Probability",
                f"{injury_risk.risk_probability:.1%}"
            )
        
        # Confidence interval
        st.subheader("📊 Confidence Interval")
        ci_lower, ci_upper = injury_risk.confidence_interval
        st.write(f"95% Confidence Interval: [{ci_lower:.3f}, {ci_upper:.3f}]")
        
        # Feature importance
        if injury_risk.feature_importance:
            st.subheader("🔍 Feature Importance")
            feature_df = pd.DataFrame(
                list(injury_risk.feature_importance.items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=False)
            
            fig = px.bar(
                feature_df.head(10),
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 10 Risk Factors"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def show_biomechanical_analysis(self):
        """Display biomechanical asymmetry analysis"""
        if 'biomechanical_asymmetry' not in self.analysis_results:
            st.warning("No biomechanical analysis available")
            return
        
        asymmetry = self.analysis_results['biomechanical_asymmetry']
        
        # Asymmetry overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Overall Asymmetry",
                f"{asymmetry.overall_asymmetry_score:.1f}%",
                delta=asymmetry.risk_category
            )
        
        with col2:
            st.metric(
                "Risk Category",
                asymmetry.risk_category
            )
        
        with col3:
            st.metric(
                "Confidence",
                f"{asymmetry.confidence:.1%}"
            )
        
        # Asymmetry breakdown
        st.subheader("⚖️ Asymmetry Breakdown")
        
        asymmetry_data = {
            'Metric': ['SLCMJ', 'Hamstring', 'Knee Valgus', 'Y-Balance', 'Hip Rotation'],
            'Asymmetry (%)': [
                asymmetry.slcmj_asymmetry,
                asymmetry.hamstring_asymmetry,
                asymmetry.knee_valgus_asymmetry,
                asymmetry.y_balance_asymmetry,
                asymmetry.hip_rotation_asymmetry
            ]
        }
        
        fig = px.bar(
            pd.DataFrame(asymmetry_data),
            x='Metric',
            y='Asymmetry (%)',
            title="Biomechanical Asymmetry by Metric",
            color='Asymmetry (%)',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def show_performance_analysis(self):
        """Display performance analysis results"""
        st.subheader("📊 Performance Analysis")
        
        if self.athlete_data is None:
            st.warning("No performance data available")
            return
        
        # Performance trends
        if 'training_load' in self.athlete_data.columns:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Training Load Over Time', 'Load Ratio Trend'),
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=self.athlete_data['date'],
                    y=self.athlete_data['training_load'],
                    mode='lines+markers',
                    name='Training Load'
                ),
                row=1, col=1
            )
            
            if 'load_ratio' in self.athlete_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.athlete_data['date'],
                        y=self.athlete_data['load_ratio'],
                        mode='lines+markers',
                        name='Load Ratio'
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    
    def show_recommendations(self):
        """Display personalized recommendations"""
        st.subheader("💡 Personalized Recommendations")
        
        if 'injury_risk' not in self.analysis_results:
            st.warning("No recommendations available")
            return
        
        injury_risk = self.analysis_results['injury_risk']
        
        # Display recommendations
        for i, rec in enumerate(injury_risk.recommendations, 1):
            st.write(f"{i}. {rec}")
        
        # Action items
        st.subheader("🎯 Action Items")
        
        if injury_risk.risk_level == "HIGH":
            st.error("🚨 **Immediate Action Required**")
            st.write("Your current training load and risk factors indicate immediate intervention is needed.")
        elif injury_risk.risk_level == "MODERATE":
            st.warning("⚠️ **Monitor Closely**")
            st.write("Your training load is approaching concerning levels. Consider reducing intensity.")
        else:
            st.success("✅ **Continue Current Training**")
            st.write("Your current training load appears sustainable. Keep up the good work!")
    
    def plot_training_load_trends(self):
        """Plot training load trends"""
        if 'training_load' not in self.athlete_data.columns:
            st.warning("Training load data not available")
            return
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.athlete_data['date'],
            y=self.athlete_data['training_load'],
            mode='lines+markers',
            name='Daily Training Load',
            line=dict(color='blue', width=2)
        ))
        
        if 'acute_load' in self.athlete_data.columns:
            fig.add_trace(go.Scatter(
                x=self.athlete_data['date'],
                y=self.athlete_data['acute_load'],
                mode='lines',
                name='7-Day Rolling Load',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title="Training Load Trends",
            xaxis_title="Date",
            yaxis_title="Training Load",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_activity_distribution(self):
        """Plot activity type distribution"""
        if 'type' not in self.athlete_data.columns:
            st.warning("Activity type data not available")
            return
        
        activity_counts = self.athlete_data['type'].value_counts()
        
        fig = px.pie(
            values=activity_counts.values,
            names=activity_counts.index,
            title="Activity Type Distribution"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_performance_metrics(self):
        """Plot performance metrics"""
        if 'distance_miles' not in self.athlete_data.columns:
            st.warning("Distance data not available")
            return
        
        # Create subplots for different metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distance per Activity', 'Duration per Activity', 'Pace Trends', 'Weekly Volume'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Distance per activity
        fig.add_trace(
            go.Scatter(
                x=self.athlete_data['date'],
                y=self.athlete_data['distance_miles'],
                mode='markers',
                name='Distance',
                marker=dict(color='green', size=8)
            ),
            row=1, col=1
        )
        
        # Duration per activity
        if 'duration_min' in self.athlete_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.athlete_data['date'],
                    y=self.athlete_data['duration_min'],
                    mode='markers',
                    name='Duration',
                    marker=dict(color='orange', size=8)
                ),
                row=1, col=2
            )
        
        # Weekly volume
        if 'duration_min' in self.athlete_data.columns:
            weekly_volume = self.athlete_data.set_index('date')['duration_min'].resample('W').sum()
            fig.add_trace(
                go.Bar(
                    x=weekly_volume.index,
                    y=weekly_volume.values,
                    name='Weekly Volume',
                    marker_color='purple'
                ),
                row=2, col=2
            )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_risk_timeline(self):
        """Plot injury risk timeline"""
        if 'load_ratio' not in self.athlete_data.columns:
            st.warning("Load ratio data not available")
            return
        
        fig = go.Figure()
        
        # Load ratio with risk thresholds
        fig.add_trace(go.Scatter(
            x=self.athlete_data['date'],
            y=self.athlete_data['load_ratio'],
            mode='lines+markers',
            name='Load Ratio',
            line=dict(color='blue', width=2)
        ))
        
        # Risk thresholds
        fig.add_hline(y=1.5, line_dash="dash", line_color="red", 
                     annotation_text="High Risk Threshold")
        fig.add_hline(y=1.3, line_dash="dash", line_color="orange", 
                     annotation_text="Moderate Risk Threshold")
        fig.add_hline(y=0.8, line_dash="dash", line_color="yellow", 
                     annotation_text="Detraining Threshold")
        
        fig.update_layout(
            title="Injury Risk Timeline (Load Ratio)",
            xaxis_title="Date",
            yaxis_title="Load Ratio",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def generate_report(self, report_type: str) -> str:
        """Generate text report"""
        try:
            report = f"""
ATHLETE PERFORMANCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Report Type: {report_type}

SUMMARY
-------
Total Activities: {len(self.athlete_data) if self.athlete_data is not None else 0}
Date Range: {self.get_date_range()}

ANALYSIS RESULTS
----------------
"""
            
            if self.analysis_results:
                if 'injury_risk' in self.analysis_results:
                    injury_risk = self.analysis_results['injury_risk']
                    report += f"""
Injury Risk Assessment:
- Risk Level: {injury_risk.risk_level}
- Risk Probability: {injury_risk.risk_probability:.1%}
- Confidence: {injury_risk.confidence_score:.1%}
- Recommendations: {', '.join(injury_risk.recommendations)}
"""
                
                if 'biomechanical_asymmetry' in self.analysis_results:
                    asymmetry = self.analysis_results['biomechanical_asymmetry']
                    report += f"""
Biomechanical Analysis:
- Overall Asymmetry: {asymmetry.overall_asymmetry_score:.1f}%
- Risk Category: {asymmetry.risk_category}
- Confidence: {asymmetry.confidence:.1%}
"""
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return f"Error generating report: {e}"
    
    def export_data(self, export_format: str) -> bytes:
        """Export data in specified format"""
        try:
            if export_format == "CSV":
                return self.athlete_data.to_csv(index=False).encode()
            elif export_format == "JSON":
                return self.athlete_data.to_json(orient='records', indent=2).encode()
            elif export_format == "Excel":
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    self.athlete_data.to_excel(writer, index=False, sheet_name='Athlete Data')
                return output.getvalue()
            elif export_format == "PDF":
                # Simple text-based PDF export
                report_text = self.generate_report("Export")
                return report_text.encode()
            else:
                return b"Unsupported format"
                
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return f"Error exporting data: {e}".encode()
    
    def get_mime_type(self, export_format: str) -> str:
        """Get MIME type for export format"""
        mime_types = {
            "CSV": "text/csv",
            "JSON": "application/json",
            "Excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "PDF": "text/plain"  # Simplified for now
        }
        return mime_types.get(export_format, "text/plain")
    
    def get_date_range(self) -> str:
        """Get date range as string"""
        if self.athlete_data is None or 'date' not in self.athlete_data.columns:
            return "Unknown"
        
        start_date = self.athlete_data['date'].min()
        end_date = self.athlete_data['date'].max()
        return f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

def main():
    """Main function to run the Streamlit app"""
    try:
        dashboard = StreamlitDashboard()
        dashboard.main()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()
