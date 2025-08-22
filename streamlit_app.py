"""
Professional Web Content Analyzer - Streamlit Interface
Clean, minimal business dashboard for website analysis
"""
import streamlit as st
import asyncio
import time
import json
from datetime import datetime
from typing import Optional
import plotly.graph_objects as go
import pandas as pd

# Configure page
st.set_page_config(
    page_title="Web Content Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    """Initialize session state variables"""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None

def render_header():
    """Render professional header section"""
    st.markdown('<div class="main-title">Web Content Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-powered website analysis and optimization insights</div>', unsafe_allow_html=True)

def render_sidebar():
    """Render configuration sidebar"""
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
        
        # Recent analyses summary
        st.header("Analysis Summary")
        total_analyses = len(st.session_state.analysis_results)
        successful = len([r for r in st.session_state.analysis_results if r.status.value == "completed"])
        total_cost = sum(getattr(r, 'cost', 0) for r in st.session_state.analysis_results)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total", total_analyses)
            st.metric("Success Rate", f"{(successful/max(total_analyses,1)*100):.0f}%")
        with col2:
            st.metric("Successful", successful)
            st.metric("Total Cost", f"${total_cost:.3f}")
        
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
    """Execute analysis with progress tracking"""
    
    try:
        # Import services
        from src.application.services.content_analysis import ContentAnalysisService
        from src.infrastructure.llm.service import ProductionLLMService
        from src.domain.models import AnalysisType
        
        # Create mock scraping proxy for interface testing
        class MockScrapingProxy:
            async def secure_scrape(self, url):
                from src.domain.models import (ScrapingResult, URLInfo, ContentMetrics, 
                                             ScrapedContent, ContentType, ScrapingStatus)
                from datetime import datetime
                
                url_info = URLInfo.from_url(url)
                content = f"This is mock scraped content for testing the interface from {url}. " * 20
                headings = ["Main Heading", "Section 1", "Section 2"]
                links = ["https://example.com/link1", "https://example.com/link2"]
                
                metrics = ContentMetrics.calculate(
                    content=content,
                    links=links,
                    headings=headings
                )
                
                scraped_content = ScrapedContent(
                    url_info=url_info,
                    title="Mock Website Title",
                    headings=headings,
                    main_content=content,
                    links=links,
                    meta_description="Mock meta description",
                    meta_keywords=["test", "mock", "content"],
                    content_type=ContentType.ARTICLE,
                    metrics=metrics,
                    scraped_at=datetime.now(),
                    status=ScrapingStatus.SUCCESS
                )
                
                return ScrapingResult(
                    request=None,
                    content=scraped_content,
                    status_code=200,
                    response_time=1.0,
                    is_success=True
                )
        
        # Initialize services
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        status_text.text("Initializing analysis...")
        progress_bar.progress(10)
        
        llm_service = ProductionLLMService()
        scraping_proxy = MockScrapingProxy()
        analysis_service = ContentAnalysisService(scraping_proxy, llm_service)
        
        status_text.text("Scraping website content...")
        progress_bar.progress(40)
        
        # Execute analysis
        analysis_type_enum = AnalysisType(analysis_type)
        result = await analysis_service.analyze_url(url, analysis_type_enum)
        
        status_text.text("Processing AI analysis...")
        progress_bar.progress(80)
        
        # Finalize
        progress_bar.progress(100)
        status_text.text("Analysis completed")
        
        # Store results
        st.session_state.current_analysis = result
        st.session_state.analysis_results.append(result)
        
        # Clear progress indicators
        time.sleep(1)
        progress_container.empty()
        
        # Show completion message
        if result.status.value == "completed":
            st.success("Analysis completed successfully")
        else:
            st.error(f"Analysis failed: {result.error_message}")
    
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")

def display_analysis_results(result):
    """Display analysis results in professional format"""
    
    if result.status.value != "completed":
        st.error(f"Analysis Status: {result.status.value}")
        if result.error_message:
            st.error(f"Error: {result.error_message}")
        return
    
    # Results header
    st.header("Analysis Results")
    
    # Key metrics overview
    if result.metrics:
        st.subheader("Performance Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Overall Score", f"{result.metrics.overall_score:.1f}/10")
        with col2:
            st.metric("Content Quality", f"{result.metrics.content_quality_score:.1f}/10")
        with col3:
            st.metric("SEO Score", f"{result.metrics.seo_score:.1f}/10")
        with col4:
            st.metric("UX Score", f"{result.metrics.ux_score:.1f}/10")
        with col5:
            st.metric("Readability", f"{result.metrics.readability_score:.1f}/10")
        
        # Performance chart
        create_performance_chart(result.metrics)
    
    # Tabbed content for detailed results
    tab1, tab2, tab3, tab4 = st.tabs(["Executive Summary", "Detailed Insights", "Recommendations", "Technical Details"])
    
    with tab1:
        st.markdown("### Executive Summary")
        st.write(result.executive_summary)
    
    with tab2:
        if result.insights:
            display_insights_section(result.insights)
        else:
            st.info("No detailed insights available")
    
    with tab3:
        if result.insights and result.insights.recommendations:
            st.markdown("### Action Items")
            for i, rec in enumerate(result.insights.recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.info("No specific recommendations available")
    
    with tab4:
        display_technical_details(result)
    
    # Export section
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    render_export_section(result)

def create_performance_chart(metrics):
    """Create clean performance visualization"""
    
    categories = ['Content', 'SEO', 'UX', 'Readability', 'Engagement']
    scores = [
        metrics.content_quality_score,
        metrics.seo_score,
        metrics.ux_score,
        metrics.readability_score,
        metrics.engagement_score
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
    
    st.plotly_chart(fig, use_container_width=True)

def display_insights_section(insights):
    """Display insights in organized format"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Strengths")
        if insights.strengths:
            for strength in insights.strengths[:5]:
                st.write(f"• {strength}")
        else:
            st.write("No specific strengths identified")
        
        st.markdown("#### Opportunities")
        if insights.opportunities:
            for opportunity in insights.opportunities[:5]:
                st.write(f"• {opportunity}")
        else:
            st.write("No opportunities identified")
    
    with col2:
        st.markdown("#### Areas for Improvement")
        if insights.weaknesses:
            for weakness in insights.weaknesses[:5]:
                st.write(f"• {weakness}")
        else:
            st.write("No major issues identified")
        
        st.markdown("#### Key Findings")
        if insights.key_findings:
            for finding in insights.key_findings[:5]:
                st.write(f"• {finding}")
        else:
            st.write("No specific findings")

def display_technical_details(result):
    """Display technical information"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Analysis Details")
        st.write(f"**Analysis ID:** `{result.analysis_id}`")
        st.write(f"**Type:** {result.analysis_type.value}")
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
            st.write(f"**Title:** {result.scraped_content.title}")
            st.write(f"**Word Count:** {result.scraped_content.metrics.word_count:,} words")
            st.write(f"**Reading Time:** {result.scraped_content.metrics.reading_time_minutes:.1f} minutes")
            st.write(f"**Analyzed:** {result.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

def render_export_section(result):
    """Render export options with strategic emoji usage"""
    
    st.subheader("📁 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export PDF", use_container_width=True):
            st.info("PDF export functionality coming soon")
    
    with col2:
        # JSON export
        export_data = create_export_data(result)
        st.download_button(
            "💾 Export JSON",
            data=json.dumps(export_data, indent=2),
            file_name=f"analysis_{result.analysis_id[:8]}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col3:
        if st.button("📊 Export CSV", use_container_width=True):
            st.info("CSV export functionality coming soon")

def create_export_data(result):
    """Create structured export data"""
    return {
        "analysis_id": result.analysis_id,
        "url": result.url,
        "analysis_type": result.analysis_type.value,
        "timestamp": result.created_at.isoformat(),
        "executive_summary": getattr(result, 'executive_summary', ''),
        "metrics": {
            "overall_score": result.metrics.overall_score,
            "content_quality": result.metrics.content_quality_score,
            "seo_score": result.metrics.seo_score,
            "ux_score": result.metrics.ux_score,
            "readability": result.metrics.readability_score
        } if result.metrics else None,
        "insights": {
            "strengths": result.insights.strengths,
            "weaknesses": result.insights.weaknesses,
            "opportunities": result.insights.opportunities,
            "recommendations": result.insights.recommendations,
            "key_findings": result.insights.key_findings
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
        history_data.append({
            "URL": result.url,
            "Type": result.analysis_type.value.replace('_', ' ').title(),
            "Status": result.status.value.title(),
            "Score": f"{result.metrics.overall_score:.1f}" if result.metrics else "N/A",
            "Cost": f"${getattr(result, 'cost', 0):.4f}",
            "Date": result.created_at.strftime('%m/%d/%Y'),
            "Time": result.created_at.strftime('%H:%M')
        })
    
    df = pd.DataFrame(history_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

def main():
    """Main application entry point"""
    
    initialize_session_state()
    render_header()
    
    # Main layout
    analysis_type, quality_preference, max_cost = render_sidebar()
    render_main_interface(analysis_type, quality_preference, max_cost)
    
    # History section
    if st.session_state.analysis_results:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        render_history_section()

if __name__ == "__main__":
    main()
