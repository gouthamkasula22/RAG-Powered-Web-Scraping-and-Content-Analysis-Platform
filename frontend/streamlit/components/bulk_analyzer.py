"""
Bulk URL Analysis Component
Allows analyzing multiple URLs simultaneously with progress tracking
"""

import streamlit as st
import asyncio
import time
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import sys

# Add backend to path
backend_path = Path(__file__).parent.parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

try:
    from src.domain.models import AnalysisType, AnalysisStatus
except ImportError:
    # Fallback enum definitions
    class AnalysisType:
        COMPREHENSIVE = "comprehensive"
        SEO_FOCUSED = "seo_focused"
        UX_FOCUSED = "ux_focused"
        CONTENT_QUALITY = "content_quality"

class BulkAnalyzer:
    """Component for bulk URL analysis with real-time progress tracking"""
    
    def __init__(self, api_base_url: str = None):
        # Configure backend URL based on environment
        if api_base_url is None:
            # Force localhost for local development
            self.api_base_url = "http://localhost:8000"
        else:
            self.api_base_url = api_base_url
        self.max_urls = 20  # Increased limit - can analyze up to 20 URLs at once
        self.max_parallel = 5  # Maximum parallel processing
        
    def create_bulk_interface(self):
        """Create the bulk analysis interface"""
        st.header("🔄 Bulk URL Analysis")
        st.markdown("Analyze multiple websites simultaneously for comprehensive insights")
        
        # Info about bulk analysis capabilities
        with st.expander("ℹ️ Bulk Analysis Information", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Capabilities:**")
                st.write(f"• Analyze up to **{self.max_urls} URLs** per batch")
                st.write(f"• **{self.max_parallel} parallel** analyses for speed")
                st.write("• Executive summaries for each site")
                st.write("• Comparative score visualization")
                st.write("• Export results to CSV/JSON/PDF")
            
            with col2:
                st.write("**Best Practices:**")
                st.write("• Use similar website types for better comparison")
                st.write("• Check all URLs are accessible before starting")
                st.write("• Allow 2-5 minutes for large batches")
                st.write("• Review failed analyses for connectivity issues")
        
        # URL input section
        urls = self._render_url_input()
        
        if not urls:
            st.info("💡 Enter URLs above to start bulk analysis")
            return
            
        # Configuration section
        config = self._render_configuration_section()
        
        # Analysis controls
        if st.button("Start Bulk Analysis", type="primary", use_container_width=True):
            if len(urls) > self.max_urls:
                st.error(f"Maximum {self.max_urls} URLs allowed per batch")
                return
                
            # Start bulk analysis
            asyncio.run(self._execute_bulk_analysis(urls, config))
            
        # Display existing bulk results if any
        self._display_bulk_results()
    
    def _render_url_input(self) -> List[str]:
        """Render URL input section with validation"""
        
        st.subheader("URLs to Analyze")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Manual Entry", "Upload CSV", "Paste List"],
            horizontal=True,
            key="bulk_input_method"
        )
        
        urls = []
        
        if input_method == "Manual Entry":
            urls = self._manual_url_entry()
        elif input_method == "Upload CSV":
            urls = self._csv_upload()
        elif input_method == "Paste List":
            urls = self._paste_urls()
            
        # Display URL validation
        if urls:
            valid_urls = []
            invalid_urls = []
            
            for url in urls:
                if self._validate_url(url):
                    valid_urls.append(url)
                else:
                    invalid_urls.append(url)
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"✅ Valid URLs: {len(valid_urls)}")
            with col2:
                if invalid_urls:
                    st.error(f"❌ Invalid URLs: {len(invalid_urls)}")
                    with st.expander("View invalid URLs"):
                        for url in invalid_urls:
                            st.text(url)
            
            return valid_urls
            
        return []
    
    def _manual_url_entry(self) -> List[str]:
        """Manual URL entry with dynamic addition"""
        
        if 'bulk_urls' not in st.session_state:
            st.session_state.bulk_urls = [""]
        
        urls = []
        
        for i, url in enumerate(st.session_state.bulk_urls):
            col1, col2 = st.columns([4, 1])
            with col1:
                new_url = st.text_input(
                    f"URL {i+1}",
                    value=url,
                    placeholder="https://example.com",
                    key=f"bulk_url_{i}",
                    label_visibility="collapsed"
                )
                if new_url.strip():
                    urls.append(new_url.strip())
                    
                # Update session state
                st.session_state.bulk_urls[i] = new_url
                
            with col2:
                if st.button("❌", key=f"remove_url_{i}", help="Remove URL"):
                    st.session_state.bulk_urls.pop(i)
                    st.rerun()
        
        # Add new URL button
        if st.button("➕ Add URL") and len(st.session_state.bulk_urls) < self.max_urls:
            st.session_state.bulk_urls.append("")
            st.rerun()
            
        return [url for url in urls if url]
    
    def _csv_upload(self) -> List[str]:
        """Upload CSV file with URLs"""
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with URLs",
            type=['csv'],
            help="CSV should have a column named 'url' or 'URLs'"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Try to find URL column
                url_column = None
                for col in df.columns:
                    if col.lower() in ['url', 'urls', 'website', 'link']:
                        url_column = col
                        break
                
                if url_column:
                    urls = df[url_column].dropna().tolist()
                    st.success(f"Found {len(urls)} URLs in '{url_column}' column")
                    return urls[:self.max_urls]  # Limit to max URLs
                else:
                    st.error("No URL column found. Please ensure your CSV has a column named 'url', 'URLs', 'website', or 'link'")
                    
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
        
        return []
    
    def _paste_urls(self) -> List[str]:
        """Paste multiple URLs"""
        
        urls_text = st.text_area(
            "Paste URLs (one per line)",
            placeholder="https://example1.com\nhttps://example2.com\nhttps://example3.com",
            height=200
        )
        
        if urls_text:
            urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
            return urls[:self.max_urls]  # Limit to max URLs
        
        return []
    
    def _validate_url(self, url: str) -> bool:
        """Basic URL validation"""
        try:
            from urllib.parse import urlparse
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _render_configuration_section(self) -> Dict[str, Any]:
        """Render analysis configuration options"""
        
        st.subheader("⚙️ Analysis Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Use radio buttons for more reliable selection
            analysis_options = [
                ("comprehensive", "Comprehensive Analysis"),
                ("seo_focused", "SEO Focused"),
                ("ux_focused", "UX Focused"),
                ("content_quality", "Content Quality")
            ]
            
            analysis_index = st.radio(
                "Analysis Type",
                options=range(len(analysis_options)),
                format_func=lambda x: analysis_options[x][1],
                key="bulk_analysis_type"
            )
            analysis_type = analysis_options[analysis_index]
        
        with col2:
            # Use radio buttons for more reliable selection
            quality_options = [
                ("balanced", "Balanced"),
                ("speed", "Speed Priority"),
                ("premium", "Premium Quality")
            ]
            
            quality_index = st.radio(
                "Quality Preference",
                options=range(len(quality_options)),
                format_func=lambda x: quality_options[x][1],
                key="bulk_quality_preference"
            )
            quality_preference = quality_options[quality_index]
        
        with col3:
            parallel_workers = st.slider(
                "Parallel Workers",
                min_value=1,
                max_value=self.max_parallel,
                value=3,
                help=f"Simultaneous analyses (max {self.max_parallel})",
                key="bulk_parallel_workers"
            )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            max_cost_per_url = st.number_input(
                "Max Cost per URL ($)",
                min_value=0.01,
                max_value=1.00,
                value=0.05,
                step=0.01,
                key="bulk_max_cost"
            )
            
            timeout_minutes = st.slider(
                "Timeout (minutes)",
                min_value=1,
                max_value=10,
                value=5,
                help="Maximum time to wait for completion",
                key="bulk_timeout"
            )
        
        return {
            "analysis_type": analysis_type[0],
            "quality_preference": quality_preference[0],
            "max_cost_per_url": max_cost_per_url,
            "parallel_limit": parallel_workers,
            "timeout_seconds": timeout_minutes * 60
        }
    
    async def _execute_bulk_analysis(self, urls: List[str], config: Dict[str, Any]):
        """Execute bulk analysis with progress tracking"""
        
        # Initialize progress tracking
        progress_container = st.container()
        results_container = st.container()
        
        with progress_container:
            st.subheader("📊 Analysis Progress")
            
            # Overall progress
            overall_progress = st.progress(0)
            status_text = st.empty()
            
            # URL-specific progress
            url_progress_data = []
            for i, url in enumerate(urls):
                url_progress_data.append({
                    "URL": url,
                    "Status": "Pending",
                    "Progress": 0,
                    "Score": "-",
                    "Cost": "-"
                })
            
            progress_df = pd.DataFrame(url_progress_data)
            progress_table = st.dataframe(progress_df, use_container_width=True)
        
        try:
            # Make API request for bulk analysis
            bulk_request = {
                "urls": urls,
                "analysis_type": config["analysis_type"],
                "quality_preference": config["quality_preference"],
                "max_cost": config["max_cost_per_url"],
                "parallel_limit": config.get("parallel_limit", 3)
            }
            
            # Start bulk analysis
            status_text.text("🚀 Starting bulk analysis...")
            
            response = requests.post(
                f"{self.api_base_url}/api/v1/analyze/bulk",
                json=bulk_request,
                timeout=config.get("timeout_seconds", 300)
            )
            
            if response.status_code != 200:
                st.error(f"Failed to start bulk analysis: {response.text}")
                return
            
            bulk_data = response.json()
            batch_id = bulk_data["batch_id"]
            
            # The bulk analysis is completed synchronously, so we have all results
            status_text.text("✅ Bulk analysis completed!")
            overall_progress.progress(100)
            
            # Update progress table with results
            updated_progress_data = []
            for result in bulk_data["results"]:
                score = "-"
                if result["metrics"] and "overall_score" in result["metrics"]:
                    score = f"{result['metrics']['overall_score']:.1f}/10"
                
                updated_progress_data.append({
                    "URL": result["url"],
                    "Status": "✅ Complete" if result["status"] == "completed" else "❌ Failed",
                    "Progress": 100 if result["status"] == "completed" else 0,
                    "Score": score,
                    "Cost": f"${result['cost']:.4f}"
                })
            
            updated_df = pd.DataFrame(updated_progress_data)
            progress_table.dataframe(updated_df, use_container_width=True)
            
            # Store results in session state
            if 'bulk_analysis_results' not in st.session_state:
                st.session_state.bulk_analysis_results = []
            
            st.session_state.bulk_analysis_results.append({
                "batch_id": batch_id,
                "timestamp": datetime.now(),
                "urls": urls,
                "config": config,
                "results": bulk_data.get("results", []),
                "total_cost": bulk_data.get("total_cost", 0),
                "completed": bulk_data.get("completed", 0),
                "failed": bulk_data.get("failed", 0)
            })
            
            # Display summary
            self._display_bulk_summary(bulk_data)
            
            # Offer to add to Knowledge Repository
            self._offer_knowledge_repository_integration(bulk_data)
            
        except Exception as e:
            st.error(f"Bulk analysis failed: {e}")
    
    def _offer_knowledge_repository_integration(self, bulk_data: Dict[str, Any]):
        """Offer to add bulk analysis results to RAG Knowledge Repository"""
        
        # Check if we have successful results
        results = bulk_data.get("results", [])
        successful_results = [r for r in results if r.get("status") == "completed"]
        
        if not successful_results:
            return
        
        st.info("💡 **Tip**: Your bulk analysis results can be added to the Knowledge Repository for intelligent Q&A!")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🧠 Add to Knowledge Repository", type="primary", help="Add these results to RAG Knowledge Repository for Q&A"):
                try:
                    # Try to import and use RAG Knowledge Repository
                    from components.rag_knowledge_repository import RAGKnowledgeRepository
                    
                    if 'rag_knowledge_repo' not in st.session_state:
                        st.session_state.rag_knowledge_repo = RAGKnowledgeRepository()
                    
                    # Load the results into knowledge base
                    rag_repo = st.session_state.rag_knowledge_repo
                    success = rag_repo._load_from_session_state()
                    
                    if success:
                        st.success("✅ Bulk analysis results added to Knowledge Repository!")
                        st.info("🔍 Visit the **Knowledge Repository** tab to ask questions about these websites.")
                    else:
                        st.warning("⚠️ Some results may already be in the knowledge base.")
                
                except ImportError:
                    st.error("❌ RAG Knowledge Repository not available. Please ensure all dependencies are installed.")
                except Exception as e:
                    st.error(f"❌ Failed to add to Knowledge Repository: {e}")
        
        with col2:
            st.markdown("**What you can do:**")
            st.markdown("• Ask questions about analyzed websites")
            st.markdown("• Compare information across multiple sites")
            st.markdown("• Get AI-powered insights and answers")

    def _display_bulk_summary(self, bulk_data: Dict[str, Any]):
        """Display enhanced bulk analysis summary with executive summaries"""
        
        st.subheader("📋 Analysis Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total URLs", bulk_data.get("total_urls", 0))
        with col2:
            st.metric("Completed", bulk_data.get("completed", 0))
        with col3:
            st.metric("Failed", bulk_data.get("failed", 0))
        with col4:
            st.metric("Total Cost", f"${bulk_data.get('total_cost', 0):.4f}")
        
        # Results breakdown
        results = bulk_data.get("results", [])
        if results:
            
            # Enhanced Score distribution chart with colors
            completed_results = [r for r in results if r.get("status") == "completed" and r.get("metrics")]
            if completed_results:
                scores = [r["metrics"].get("overall_score", 0) for r in completed_results]
                urls = [r["url"][:30] + "..." if len(r["url"]) > 30 else r["url"] for r in completed_results]
                
                # Create a colored bar chart instead of histogram
                fig = go.Figure(data=[
                    go.Bar(
                        x=urls,
                        y=scores,
                        marker=dict(
                            color=scores,
                            colorscale='RdYlGn',  # Red-Yellow-Green scale
                            cmin=0,
                            cmax=10,
                            colorbar=dict(title="Score")
                        ),
                        text=[f"{score:.1f}/10" for score in scores],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Overall Scores by Website",
                    xaxis_title="Website",
                    yaxis_title="Overall Score",
                    xaxis_tickangle=-45,
                    yaxis=dict(range=[0, 10]),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Executive Summaries Section
            st.subheader("📝 Executive Summaries")
            
            for i, result in enumerate(completed_results):
                if result.get("executive_summary"):
                    with st.expander(f"📄 {result['url'][:50]}{'...' if len(result['url']) > 50 else ''} (Score: {result['metrics'].get('overall_score', 0):.1f}/10)"):
                        st.write("**Executive Summary:**")
                        st.write(result["executive_summary"])
                        
                        # Show key metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Content Quality", f"{result['metrics'].get('content_quality_score', 0):.1f}/10")
                        with col2:
                            st.metric("SEO Score", f"{result['metrics'].get('seo_score', 0):.1f}/10")
                        with col3:
                            st.metric("UX Score", f"{result['metrics'].get('ux_score', 0):.1f}/10")
                        
                        # Show key insights
                        if result.get("insights"):
                            st.write("**Key Strengths:**")
                            for strength in result["insights"].get("strengths", [])[:3]:
                                st.write(f"• {strength}")
                            
                            st.write("**Top Recommendations:**")
                            for rec in result["insights"].get("recommendations", [])[:3]:
                                st.write(f"• {rec}")
            
            # Detailed Results Table
            st.subheader("📊 Detailed Results")
            
            results_data = []
            for result in results:
                status_emoji = "✅" if result.get("status") == "completed" else "❌"
                results_data.append({
                    "Status": f"{status_emoji} {result['status'].title()}",
                    "URL": result["url"],
                    "Overall Score": f"{result.get('metrics', {}).get('overall_score', 0):.1f}/10" if result.get('metrics') else "N/A",
                    "Content Quality": f"{result.get('metrics', {}).get('content_quality_score', 0):.1f}/10" if result.get('metrics') else "N/A",
                    "SEO Score": f"{result.get('metrics', {}).get('seo_score', 0):.1f}/10" if result.get('metrics') else "N/A",
                    "UX Score": f"{result.get('metrics', {}).get('ux_score', 0):.1f}/10" if result.get('metrics') else "N/A",
                    "Cost": f"${result.get('cost', 0):.4f}",
                    "Time": f"{result.get('processing_time', 0):.1f}s",
                    "Error": result.get('error_message', '') if result.get('status') != 'completed' else ''
                })
            
            if results_data:
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Export options
                self._render_bulk_export_options(results_df, bulk_data)
    
    def _render_bulk_export_options(self, results_df: pd.DataFrame, bulk_data: Dict[str, Any]):
        """Render export options for bulk results"""
        
        st.subheader("📤 Export Results")
        
        # Generate export data once
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create enhanced CSV with more details
        enhanced_csv_data = []
        for result in bulk_data.get("results", []):
            row = {
                "URL": result["url"],
                "Status": result["status"],
                "Overall_Score": result.get('metrics', {}).get('overall_score', 0) if result.get('metrics') else 0,
                "Content_Quality_Score": result.get('metrics', {}).get('content_quality_score', 0) if result.get('metrics') else 0,
                "SEO_Score": result.get('metrics', {}).get('seo_score', 0) if result.get('metrics') else 0,
                "UX_Score": result.get('metrics', {}).get('ux_score', 0) if result.get('metrics') else 0,
                "Readability_Score": result.get('metrics', {}).get('readability_score', 0) if result.get('metrics') else 0,
                "Engagement_Score": result.get('metrics', {}).get('engagement_score', 0) if result.get('metrics') else 0,
                "Cost_USD": result.get('cost', 0),
                "Processing_Time_Seconds": result.get('processing_time', 0),
                "Provider_Used": result.get('provider_used', ''),
                "Executive_Summary": result.get('executive_summary', ''),
                "Error_Message": result.get('error_message', '') if result.get('status') != 'completed' else ''
            }
            
            # Add insights as separate columns
            if result.get('insights'):
                row["Top_Strengths"] = " | ".join(result['insights'].get('strengths', [])[:3])
                row["Key_Weaknesses"] = " | ".join(result['insights'].get('weaknesses', [])[:3])
                row["Top_Recommendations"] = " | ".join(result['insights'].get('recommendations', [])[:3])
                row["Key_Findings"] = " | ".join(result['insights'].get('key_findings', [])[:3])
            
            enhanced_csv_data.append(row)
        
        enhanced_df = pd.DataFrame(enhanced_csv_data)
        csv_data = enhanced_df.to_csv(index=False)
        
        # Prepare JSON data with metadata
        export_json = {
            "export_metadata": {
                "export_timestamp": timestamp,
                "total_urls": bulk_data.get("total_urls", 0),
                "completed": bulk_data.get("completed", 0),
                "failed": bulk_data.get("failed", 0),
                "total_cost": bulk_data.get("total_cost", 0),
                "batch_id": bulk_data.get("batch_id", "unknown")
            },
            "analysis_results": bulk_data.get("results", []),
            "summary_statistics": {
                "average_score": sum(r.get('metrics', {}).get('overall_score', 0) for r in bulk_data.get("results", []) if r.get('metrics')) / max(len([r for r in bulk_data.get("results", []) if r.get('metrics')]), 1),
                "score_distribution": {
                    "excellent": len([r for r in bulk_data.get("results", []) if r.get('metrics', {}).get('overall_score', 0) >= 8]),
                    "good": len([r for r in bulk_data.get("results", []) if 6 <= r.get('metrics', {}).get('overall_score', 0) < 8]),
                    "needs_improvement": len([r for r in bulk_data.get("results", []) if r.get('metrics', {}).get('overall_score', 0) < 6])
                }
            }
        }
        json_data = json.dumps(export_json, indent=2, default=str)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="📊 Download Enhanced CSV",
                data=csv_data,
                file_name=f"bulk_analysis_enhanced_{timestamp}.csv",
                mime="text/csv",
                help="Download detailed CSV with all metrics and insights",
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                label="📋 Download Complete JSON",
                data=json_data,
                file_name=f"bulk_analysis_complete_{timestamp}.json",
                mime="application/json",
                help="Download complete analysis data in JSON format",
                use_container_width=True
            )
        
        with col3:
            # Generate a simple PDF report
            pdf_data = self._generate_bulk_pdf_report(bulk_data, timestamp)
            if pdf_data:
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_data,
                    file_name=f"bulk_analysis_report_{timestamp}.pdf",
                    mime="application/pdf",
                    help="Download summary PDF report",
                    use_container_width=True
                )
            else:
                st.button("📈 PDF Report", disabled=True, help="PDF generation requires reportlab", use_container_width=True)
        
        # Additional export info
        st.markdown("---")
        with st.expander("📁 Export Details", expanded=False):
            st.write("**File Contents:**")
            st.write("• **Enhanced CSV**: All scores, executive summaries, insights, and metadata in spreadsheet format")
            st.write("• **Complete JSON**: Full analysis data with statistics and metadata for programmatic use")
            st.write("• **PDF Report**: Professional summary report with charts and tables for sharing")
            
            st.write("**File Locations:**")
            st.write("Files are downloaded to your browser's default download folder")
            st.write(f"**Timestamp**: {timestamp} (YYYYMMDD_HHMMSS format)")
    
    def _generate_bulk_pdf_report(self, bulk_data: Dict[str, Any], timestamp: str) -> Optional[bytes]:
        """Generate a PDF report for bulk analysis results"""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from io import BytesIO
            
            # Create PDF buffer
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            
            # Get styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                textColor=colors.darkblue,
                alignment=1  # Center alignment
            )
            
            # Build PDF content
            content = []
            
            # Title
            content.append(Paragraph("Bulk Website Analysis Report", title_style))
            content.append(Spacer(1, 20))
            
            # Summary section
            summary_data = [
                ["Metric", "Value"],
                ["Total URLs Analyzed", str(bulk_data.get("total_urls", 0))],
                ["Successfully Completed", str(bulk_data.get("completed", 0))],
                ["Failed Analyses", str(bulk_data.get("failed", 0))],
                ["Total Cost", f"${bulk_data.get('total_cost', 0):.4f}"],
                ["Analysis Date", timestamp[:8]]  # YYYYMMDD
            ]
            
            summary_table = Table(summary_data, colWidths=[2*inch, 2*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            content.append(Paragraph("Analysis Summary", styles['Heading2']))
            content.append(summary_table)
            content.append(Spacer(1, 20))
            
            # Results table
            results_data = [["URL", "Status", "Overall Score", "Content", "SEO", "UX"]]
            
            for result in bulk_data.get("results", [])[:15]:  # Limit to first 15 for PDF
                url_short = result["url"][:40] + "..." if len(result["url"]) > 40 else result["url"]
                status = "✓" if result.get("status") == "completed" else "✗"
                
                if result.get("metrics"):
                    overall = f"{result['metrics'].get('overall_score', 0):.1f}"
                    content_score = f"{result['metrics'].get('content_quality_score', 0):.1f}"
                    seo_score = f"{result['metrics'].get('seo_score', 0):.1f}"
                    ux_score = f"{result['metrics'].get('ux_score', 0):.1f}"
                else:
                    overall = content_score = seo_score = ux_score = "N/A"
                
                results_data.append([url_short, status, overall, content_score, seo_score, ux_score])
            
            results_table = Table(results_data, colWidths=[2.5*inch, 0.5*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
            results_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
            ]))
            
            content.append(Paragraph("Detailed Results", styles['Heading2']))
            content.append(results_table)
            
            if len(bulk_data.get("results", [])) > 15:
                content.append(Spacer(1, 10))
                content.append(Paragraph(f"Note: Showing first 15 results. Total: {len(bulk_data.get('results', []))} URLs analyzed.", styles['Normal']))
            
            # Build PDF
            doc.build(content)
            buffer.seek(0)
            return buffer.getvalue()
            
        except ImportError:
            return None
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
            return None

    def _display_bulk_results(self):
        """Display previous bulk analysis results"""
        
        if 'bulk_analysis_results' not in st.session_state or not st.session_state.bulk_analysis_results:
            return
        
        st.subheader("📚 Previous Bulk Analyses")
        
        for i, bulk_result in enumerate(reversed(st.session_state.bulk_analysis_results)):
            with st.expander(f"Bulk Analysis {i+1} - {bulk_result['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("URLs Analyzed", len(bulk_result['urls']))
                with col2:
                    st.metric("Completed", bulk_result['completed'])
                with col3:
                    st.metric("Total Cost", f"${bulk_result['total_cost']:.4f}")
                
                # Show URLs
                st.write("**URLs:**")
                for url in bulk_result['urls']:
                    st.text(f"• {url}")
                
                # Results summary
                if bulk_result['results']:
                    avg_score = sum(r.get('metrics', {}).get('overall_score', 0) for r in bulk_result['results']) / len(bulk_result['results'])
                    st.metric("Average Score", f"{avg_score:.1f}/10")


# Usage example for testing
if __name__ == "__main__":
    bulk_analyzer = BulkAnalyzer()
    bulk_analyzer.create_bulk_interface()
