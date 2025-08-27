"""
Enhanced Web Content Analyzer - Streamlit Interface
WBS 2.4: Professional interface with advanced features and responsive design
"""
import streamlit as st
import asyncio
import time
import json
import sys
import os
import requests
from datetime import datetime
from typing import Optional
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

# Add backend to Python path
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))


try:
    from components.progress_tracker import AnalysisProgressTracker, BackgroundTaskManager
    from components.report_navigator import ReportNavigator, ReportComparison
    from components.history_manager import AnalysisHistoryManager
    from components.bulk_analyzer import BulkAnalyzer
    from components.rag_knowledge_repository import RAGKnowledgeRepository
    from components.knowledge_repository import IntelligentKnowledgeRepository
    from utils.responsive_layout import ResponsiveLayout, SessionStateManager
except ImportError as e:
    st.error(f"Failed to import components: {e}")
    st.stop()

# Initialize component instances
responsive_layout = ResponsiveLayout()
session_manager = SessionStateManager()
history_manager = AnalysisHistoryManager()
bulk_analyzer = BulkAnalyzer()

# Try to use RAG Knowledge Repository, fallback to old one if it fails
try:
    knowledge_repository = RAGKnowledgeRepository()
    st.session_state.using_rag = True
except Exception as e:
    st.warning(f"RAG Knowledge Repository not available, using fallback: {e}")
    knowledge_repository = IntelligentKnowledgeRepository()
    st.session_state.using_rag = False

# Configure page
st.set_page_config(
    page_title="Web Content Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject responsive CSS
responsive_layout.inject_responsive_css()

# Professional CSS styling
st.markdown("""
<style>
    /* Remove default Streamlit styling */
    .stApp > header {
        background-color: transparent;
    }
    
    /* Main content styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #666666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Clean metric cards */
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 4px;
        border: 1px solid #e9ecef;
    }
    
    /* Status indicators */
    .status-success {
        color: #28a745;
        font-weight: 500;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: 500;
    }
    
    .status-pending {
        color: #6c757d;
        font-weight: 500;
    }
    
    /* Subtle section separators */
    .section-divider {
        border-top: 1px solid #e9ecef;
        margin: 2rem 0 1.5rem 0;
    }
    
    /* Clean button styling */
    .stButton > button {
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #0056b3;
    }
    
    /* Clean tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f8f9fa;
        border-radius: 4px 4px 0px 0px;
        color: #495057;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #007bff;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize enhanced session state variables"""
    session_manager.init_base_state()
    
    # Legacy compatibility
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None

def render_header():
    """Render professional header section"""
    st.markdown('<div class="main-title">Web Content Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-powered website analysis and optimization insights</div>', unsafe_allow_html=True)

def render_sidebar():
    """Render enhanced configuration sidebar"""
    
    background_task_manager = BackgroundTaskManager()
    
    with st.sidebar:
        st.header("Analysis Configuration")
        
        # Analysis type
        analysis_type = st.selectbox(
            "Analysis Type",
            options=[
                ("comprehensive", "Comprehensive Analysis"),
                ("seo_focused", "SEO Analysis"),
                ("ux_focused", "UX Analysis"),
                ("content_quality", "Content Quality")
            ],
            format_func=lambda x: x[1]
        )
        
        # Quality preference
        quality_preference = st.selectbox(
            "Processing Mode",
            options=[
                ("balanced", "Balanced (Free + Premium)"),
                ("speed", "Fast (Free Only)"),
                ("premium", "Premium (Best Quality)")
            ],
            format_func=lambda x: x[1]
        )
        
        # Cost limit
        max_cost = st.slider(
            "Max Cost per Analysis",
            min_value=0.00,
            max_value=0.20,
            value=0.05,
            step=0.01,
            format="$%.2f"
        )
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Enhanced metrics summary
        st.subheader("Analysis Summary")
        
        # Get statistics from history manager
        stats = history_manager.get_statistics()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total", stats.get('total_analyses', 0))
            success_rate = (stats.get('successful_analyses', 0) / max(stats.get('total_analyses', 1), 1)) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        with col2:
            st.metric("Successful", stats.get('successful_analyses', 0))
            st.metric("Total Cost", f"${stats.get('total_cost', 0):.3f}")
        
        # Background tasks display
        background_task_manager.render_active_tasks()
        
        return analysis_type, quality_preference, max_cost

def render_main_interface(analysis_type, quality_preference, max_cost):
    """Render main analysis interface"""
    
    # Input section
    st.header("Website Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        url_input = st.text_input(
            "Website URL",
            placeholder="https://example.com",
            label_visibility="collapsed"
        )
    
    with col2:
        analyze_button = st.button("Analyze", type="primary", use_container_width=True)
    
    # Handle analysis
    if analyze_button and url_input:
        asyncio.run(run_analysis(url_input, analysis_type[0], quality_preference[0], max_cost))
    elif analyze_button and not url_input:
        st.error("Please enter a valid URL")
    
    # Display current results
    if st.session_state.current_analysis:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        display_analysis_results(st.session_state.current_analysis)

async def run_analysis(url: str, analysis_type: str, quality_preference: str, max_cost: float):
    """Execute analysis with enhanced progress tracking"""
    
    try:
        # Create enhanced progress tracker
        progress_tracker = AnalysisProgressTracker()
        ui_components = progress_tracker.create_progress_interface()
        
        # Start tracking
        progress_tracker.start_tracking()
        
        # Initialize progress
        progress_tracker.update_progress(ui_components, 0, 10)
        
        # Use API call for analysis
        import requests
        import json
        
        # Update progress - validation stage
        progress_tracker.update_progress(ui_components, 0, 50)
        
        # Prepare API request
        api_url = "http://localhost:8000/api/analyze"
        request_data = {
            "url": url,
            "analysis_type": analysis_type.lower(),
            "quality_preference": quality_preference.lower(),
            "max_cost": max_cost
        }
        
        # Update progress - sending request
        progress_tracker.update_progress(ui_components, 1, 20)
        
        # Execute analysis via API
        try:
            response = requests.post(api_url, json=request_data, timeout=120)
            response.raise_for_status()
            
            # Update progress - processing
            progress_tracker.update_progress(ui_components, 2, 70)
            
            result_data = response.json()
            
            # Update progress - generating report
            progress_tracker.update_progress(ui_components, 3, 90)
            
            # Create a simple result object without backend imports
            from datetime import datetime
            
            class SimpleAnalysisResult:
                def __init__(self, data):
                    self.url = data.get("url", "")
                    self.analysis_id = data.get("analysis_id", "")
                    self.analysis_type = data.get("analysis_type", "comprehensive")
                    self.status = data.get("status", "completed")
                    self.executive_summary = data.get("executive_summary", "")
                    self.metrics = data.get("metrics", {})
                    self.insights = data.get("insights", {})
                    self.scraped_content = data.get("scraped_content", {})  # Add scraped content
                    self.processing_time = data.get("processing_time", 0.0)
                    self.cost = data.get("cost", 0.0)
                    self.provider_used = data.get("provider_used", "")
                    self.created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now()
                    self.error_message = data.get("error_message")
            
            result = SimpleAnalysisResult(result_data)
            
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to backend API: {e}")
            st.info("Make sure the backend service is running on http://localhost:8000")
            return
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.error(f"Error details: {str(e)}")
            return
        
        # Complete progress
        progress_tracker.update_progress(ui_components, 4, 100)
        
        # Store results with duplication guard
        st.session_state.current_analysis = result
        existing_ids = {getattr(r, 'analysis_id', None) for r in st.session_state.analysis_results}
        if getattr(result, 'analysis_id', None) not in existing_ids:
            st.session_state.analysis_results.append(result)
        
        # Save to persistent history (avoid duplicate insert for same analysis id in rapid reruns)
        if getattr(result, 'analysis_id', None):
            history_manager.save_analysis(result)
        
        # Clear progress indicators after brief delay
        time.sleep(1)
        st.rerun()
        
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return

def display_analysis_results(result):
    """Display enhanced analysis results with navigation and search"""
    
    # Handle both string and enum status
    status_value = result.status if isinstance(result.status, str) else result.status.value
    
    if status_value != "completed":
        st.error(f"Analysis Status: {status_value}")
        if result.error_message:
            st.error(f"Error: {result.error_message}")
            
            # Provide helpful suggestions based on error type
            if "503" in result.error_message or "Service unavailable" in result.error_message:
                st.warning("💡 **Suggestion**: E-commerce sites like Amazon often block automated requests. Try these alternatives:")
                st.markdown("""
                - Test with simpler websites: `https://example.com` or `https://httpbin.org/html`
                - Try news sites or blogs instead of e-commerce platforms
                - Wait a few minutes before retrying the same URL
                """)
            elif "403" in result.error_message or "forbidden" in result.error_message:
                st.info("💡 **Suggestion**: This website is blocking access. Try a different URL or contact the site owner.")
            elif "timeout" in result.error_message.lower():
                st.info("💡 **Suggestion**: The request timed out. Try again or check if the website is responding properly.")
            elif "404" in result.error_message:
                st.info("💡 **Suggestion**: The URL was not found. Please check the URL and try again.")
        return
    
    # Create report navigator (single creation per rerun; structural duplication removed elsewhere)
    report_navigator = ReportNavigator()
    nav_interface = report_navigator.create_navigation_interface(result)
    search_query = nav_interface["search_query"]
    section_filter = nav_interface["section_filter"]
    view_mode = nav_interface["view_mode"]
    
    # Handle search
    if search_query:
        search_results = report_navigator.search_content(result, search_query)
        with nav_interface["content_container"]:
            report_navigator.render_search_results(search_results, search_query)
        return
    
    # Results header
    st.header("Analysis Results")
    
    # Enhanced metrics display using responsive layout
    if result.metrics:
        st.subheader("Performance Overview")
        
        # Helper function to safely get metric values
        def get_metric_value(metrics, key, default=0):
            if isinstance(metrics, dict):
                return metrics.get(key, default)
            else:
                return getattr(metrics, key, default)
        
        # Use responsive metrics layout
        metrics_data = {
            "Overall Score": f"{get_metric_value(result.metrics, 'overall_score'):.1f}/10",
            "Content Quality": f"{get_metric_value(result.metrics, 'content_quality_score'):.1f}/10",
            "SEO Score": f"{get_metric_value(result.metrics, 'seo_score'):.1f}/10",
            "UX Score": f"{get_metric_value(result.metrics, 'ux_score'):.1f}/10",
            "Readability": f"{get_metric_value(result.metrics, 'readability_score'):.1f}/10"
        }
        
        responsive_layout.create_mobile_friendly_metrics(metrics_data)
        
        # Performance chart
        chart_id = getattr(result, 'analysis_id', None)
        if not chart_id:
            chart_id = hex(abs(hash(str(result))) % 0xFFFFFF)[2:]
        create_performance_chart(result.metrics, chart_key=f"performance_chart_{chart_id}")
    
    # Enhanced tabbed content with icons
    tabs = responsive_layout.create_responsive_tabs(
        ["Executive Summary", "Detailed Insights", "Recommendations", "Technical Details"],
        ["📋", "🔍", "💡", "⚙️"]
    )
    
    with tabs[0]:
        st.markdown("### Executive Summary")
        st.write(result.executive_summary)
    
    with tabs[1]:
        if result.insights:
            display_insights_section(result.insights)
        else:
            st.info("No detailed insights available")
    
    with tabs[2]:
        # Safe insights access
        recommendations = []
        if result.insights:
            if isinstance(result.insights, dict):
                recommendations = result.insights.get('recommendations', [])
            else:
                recommendations = getattr(result.insights, 'recommendations', [])
        
        if recommendations:
            st.markdown("### Action Items")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.info("No specific recommendations available")
    
    with tabs[3]:
        display_technical_details(result)
    
    # Report navigation sidebar (temporarily disabled to fix attribute errors)
    # outline = report_navigator.create_report_outline(result)
    # report_navigator.render_navigation_sidebar(outline)
    
    # Export section
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    render_export_section(result)

def create_performance_chart(metrics, chart_key: str):
    """Create clean performance visualization with unique key to avoid duplicate element id."""
    # Guard: prevent accidental double rendering within same rerun
    rendered_key_set = st.session_state.setdefault('_rendered_charts', set())
    if chart_key in rendered_key_set:
        return
    
    # Helper function to safely get metric values
    def get_metric_value(metrics, key, default=0):
        if isinstance(metrics, dict):
            return metrics.get(key, default)
        else:
            return getattr(metrics, key, default)
    
    categories = ['Content', 'SEO', 'UX', 'Readability', 'Engagement']
    scores = [
        get_metric_value(metrics, 'content_quality_score'),
        get_metric_value(metrics, 'seo_score'),
        get_metric_value(metrics, 'ux_score'),
        get_metric_value(metrics, 'readability_score'),
        get_metric_value(metrics, 'engagement_score')
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        fillcolor='rgba(0, 123, 255, 0.1)',
        line=dict(color='rgb(0, 123, 255)', width=2),
        marker=dict(color='rgb(0, 123, 255)', size=6),
        name='Performance Scores'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickfont=dict(size=10),
                gridcolor='rgba(0,0,0,0.1)'
            ),
            angularaxis=dict(
                tickfont=dict(size=11)
            )
        ),
        showlegend=False,
        width=400,
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True, key=chart_key)
    rendered_key_set.add(chart_key)

def display_insights_section(insights):
    """Display insights in organized format"""
    
    # Helper function to safely get insight values
    def get_insight_value(insights, key, default=None):
        if not insights:
            return default if default is not None else []
        if isinstance(insights, dict):
            return insights.get(key, default if default is not None else [])
        else:
            return getattr(insights, key, default if default is not None else [])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Strengths")
        strengths = get_insight_value(insights, 'strengths')
        if strengths:
            for strength in strengths[:5]:
                st.write(f"• {strength}")
        else:
            st.write("No specific strengths identified")
        
        st.markdown("#### Opportunities")
        opportunities = get_insight_value(insights, 'opportunities')
        if opportunities:
            for opportunity in opportunities[:5]:
                st.write(f"• {opportunity}")
        else:
            st.write("No opportunities identified")
    
    with col2:
        st.markdown("#### Areas for Improvement")
        weaknesses = get_insight_value(insights, 'weaknesses')
        if weaknesses:
            for weakness in weaknesses[:5]:
                st.write(f"• {weakness}")
        else:
            st.write("No major issues identified")
        
        st.markdown("#### Key Findings")
        key_findings = get_insight_value(insights, 'key_findings')
        if key_findings:
            for finding in key_findings[:5]:
                st.write(f"• {finding}")
        else:
            st.write("No specific findings")

def display_technical_details(result):
    """Display technical information"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Analysis Details")
        st.write(f"**Analysis ID:** `{result.analysis_id}`")
        # Handle both string and enum analysis_type
        analysis_type_value = result.analysis_type if isinstance(result.analysis_type, str) else result.analysis_type.value
        st.write(f"**Type:** {analysis_type_value}")
        st.write(f"**Provider:** {getattr(result, 'provider_used', 'Unknown')}")
        st.write(f"**Processing Time:** {getattr(result, 'processing_time', 0):.2f} seconds")
        st.write(f"**Cost:** ${getattr(result, 'cost', 0):.4f}")
    
    with col2:
        if hasattr(result, 'content_metrics') and result.content_metrics:
            st.markdown("#### Content Details")
            st.write(f"**Word Count:** {result.content_metrics.word_count:,} words")
            st.write(f"**Reading Time:** {result.content_metrics.reading_time_minutes:.1f} minutes")
            st.write(f"**Paragraphs:** {result.content_metrics.paragraph_count}")
            st.write(f"**Analyzed:** {getattr(result, 'analyzed_at', result.created_at).strftime('%Y-%m-%d %H:%M:%S')}")
        elif hasattr(result, 'scraped_content') and result.scraped_content:
            st.markdown("#### Content Details")
            # Handle scraped_content as dictionary
            if isinstance(result.scraped_content, dict):
                st.write(f"**Title:** {result.scraped_content.get('title', 'N/A')}")
                st.write(f"**Word Count:** {result.scraped_content.get('word_count', 0):,} words")
                st.write(f"**URL:** {result.scraped_content.get('url', 'N/A')}")
            else:
                # Handle scraped_content as object (legacy)
                st.write(f"**Title:** {getattr(result.scraped_content, 'title', 'N/A')}")
                st.write(f"**Word Count:** {getattr(result.scraped_content.metrics, 'word_count', 0):,} words")
                st.write(f"**Reading Time:** {getattr(result.scraped_content.metrics, 'reading_time_minutes', 0):.1f} minutes")
            st.write(f"**Analyzed:** {result.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

def render_export_section(result):
    """Render export options with full functionality"""
    
    st.subheader("📁 Export Results")
    
    # Create report navigator instance for export functionality
    report_navigator = ReportNavigator()
    
    # Use the full export interface
    report_navigator.create_export_options(result)

def create_export_data(result):
    """Create structured export data"""
    # Handle both string and enum analysis_type
    analysis_type_value = result.analysis_type if isinstance(result.analysis_type, str) else result.analysis_type.value
    
    # Helper function to safely get metric values
    def get_metric_value(metrics, key, default=0):
        if not metrics:
            return default
        if isinstance(metrics, dict):
            return metrics.get(key, default)
        else:
            return getattr(metrics, key, default)
    
    # Helper function to safely get insight values
    def get_insight_value(insights, key, default=None):
        if not insights:
            return default
        if isinstance(insights, dict):
            return insights.get(key, default if default is not None else [])
        else:
            return getattr(insights, key, default if default is not None else [])
    
    return {
        "analysis_id": result.analysis_id,
        "url": result.url,
        "analysis_type": analysis_type_value,
        "timestamp": result.created_at.isoformat(),
        "executive_summary": getattr(result, 'executive_summary', ''),
        "metrics": {
            "overall_score": get_metric_value(result.metrics, 'overall_score'),
            "content_quality": get_metric_value(result.metrics, 'content_quality_score'),
            "seo_score": get_metric_value(result.metrics, 'seo_score'),
            "ux_score": get_metric_value(result.metrics, 'ux_score'),
            "readability": get_metric_value(result.metrics, 'readability_score')
        } if result.metrics else None,
        "insights": {
            "strengths": get_insight_value(result.insights, 'strengths'),
            "weaknesses": get_insight_value(result.insights, 'weaknesses'),
            "opportunities": get_insight_value(result.insights, 'opportunities'),
            "recommendations": get_insight_value(result.insights, 'recommendations'),
            "key_findings": get_insight_value(result.insights, 'key_findings')
        } if result.insights else None,
        "technical_details": {
            "provider": getattr(result, 'provider_used', 'Unknown'),
            "processing_time": getattr(result, 'processing_time', 0),
            "cost": getattr(result, 'cost', 0)
        }
    }

def render_history_section():
    """Render analysis history"""
    
    if not st.session_state.analysis_results:
        st.info("No analysis history available")
        return
    
    st.header("📈 Analysis History")
    
    # Create history dataframe
    history_data = []
    for result in reversed(st.session_state.analysis_results):
        # Handle both string and enum types
        analysis_type_value = result.analysis_type if isinstance(result.analysis_type, str) else result.analysis_type.value
        status_value = result.status if isinstance(result.status, str) else result.status.value
        
        # Safe metric access
        if result.metrics:
            if isinstance(result.metrics, dict):
                overall_score = result.metrics.get('overall_score', 0)
            else:
                overall_score = getattr(result.metrics, 'overall_score', 0)
            score_display = f"{overall_score:.1f}"
        else:
            score_display = "N/A"
        
        history_data.append({
            "URL": result.url,
            "Type": analysis_type_value.replace('_', ' ').title(),
            "Status": status_value.title(),
            "Score": score_display,
            "Cost": f"${getattr(result, 'cost', 0):.4f}",
            "Date": result.created_at.strftime('%m/%d/%Y'),
            "Time": result.created_at.strftime('%H:%M')
        })
    
    df = pd.DataFrame(history_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

def main():
    """Enhanced main application entry point"""
    
    initialize_session_state()
    render_header()
    
    # Create page navigation
    page = st.selectbox(
        "Navigation",
        options=["Analysis", "Bulk Analysis", "Knowledge Repository", "History", "Comparison"],
        index=0,
        key="main_navigation"
    )
    
    if page == "Analysis":
        # Main analysis interface
        analysis_type, quality_preference, max_cost = render_sidebar()
        render_main_interface(analysis_type, quality_preference, max_cost)
    
    elif page == "Bulk Analysis":
        # Bulk analysis interface
        bulk_analyzer.create_bulk_interface()
        
    elif page == "Knowledge Repository":
        # RAG-based Knowledge Repository interface
        try:
            from components.rag_knowledge_repository import RAGKnowledgeRepository
            
            # Initialize RAG Knowledge Repository
            if 'rag_knowledge_repo' not in st.session_state:
                st.session_state.rag_knowledge_repo = RAGKnowledgeRepository()
            
            # Render the RAG interface
            st.session_state.rag_knowledge_repo.render()
            
        except ImportError as e:
            st.error(f"RAG Knowledge Repository not available: {e}")
            st.info("Please install required dependencies: pip install sentence-transformers")
            
            # Fallback to old knowledge repository
            knowledge_repository.render_main_interface()
        except Exception as e:
            st.error(f"Error initializing RAG Knowledge Repository: {e}")
            st.info("Falling back to standard Knowledge Repository...")
            
            # Fallback to old knowledge repository
            knowledge_repository.render_main_interface()
    
    elif page == "History":
        # Enhanced history interface
        history_manager.render_history_interface()
    
    elif page == "Comparison":
        # Report comparison interface
        
        # Load all available analyses (both session and database)
        available_analyses = list(st.session_state.analysis_results)  # Current session analyses
        
        # Also load recent analyses from database
        db_analyses = history_manager.load_analyses(limit=50)
        
        # Convert database records to analysis objects for comparison
        for db_analysis in db_analyses:
            # Create a simple analysis object from database record
            class DatabaseAnalysis:
                def __init__(self, record):
                    self.analysis_id = record['id']
                    self.url = record['url']
                    self.created_at = datetime.fromisoformat(record['created_at']) if record['created_at'] else datetime.now()
                    self.analysis_type = record['analysis_type']
                    self.status = record['status']
                    # Create metrics object
                    self.metrics = type('Metrics', (), {
                        'overall_score': record.get('overall_score', 0.0),
                        'content_quality_score': record.get('content_quality_score', 0.0),
                        'seo_score': record.get('seo_score', 0.0),
                        'ux_score': record.get('ux_score', 0.0)
                    })()
                    self.executive_summary = record.get('executive_summary', '')
                    self.cost = record.get('cost', 0.0)
                    self.processing_time = record.get('processing_time', 0.0)
                    self.provider_used = record.get('provider_used', '')
            
            # Only add if not already in session (avoid duplicates)
            if not any(getattr(a, 'analysis_id', None) == db_analysis['id'] for a in available_analyses):
                available_analyses.append(DatabaseAnalysis(db_analysis))
        
        if len(available_analyses) >= 2:
            report_comparison = ReportComparison()
            report_comparison.create_comparison_interface(available_analyses)
        else:
            st.info(f"Need at least 2 analyses for comparison. Found {len(available_analyses)} analyses total (session: {len(st.session_state.analysis_results)}, database: {len(db_analyses)}). Complete some analyses first.")

if __name__ == "__main__":
    main()
