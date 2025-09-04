"""
Interactive Report Navigator with Search Functionality
WBS 2.4: Advanced report viewing and navigation
"""

# Standard library imports
import json
import re
from datetime import datetime
from typing import Dict, List

# Third-party imports
import pandas as pd
import streamlit as st

class ReportNavigator:
    """Interactive report navigation with search and filtering"""

    def __init__(self):
        self.search_index = {}
        self.current_filter = None

    def create_navigation_interface(self, analysis_result) -> Dict:
        """Create interactive navigation interface"""
        # Create layout containers
        nav_container = st.container()
        content_container = st.container()

        with nav_container:
            # Stable keys per analysis id (increment version only when analysis id changes)
            analysis_id = getattr(analysis_result, 'analysis_id', str(hash(str(analysis_result))))
            prev_id = st.session_state.get('_nav_current_analysis_id')
            if prev_id != analysis_id:
                st.session_state['_nav_current_analysis_id'] = analysis_id
                st.session_state['nav_version'] = st.session_state.get('nav_version', 0) + 1
            nav_version = st.session_state.get('nav_version', 1)
            suffix = f"_v{nav_version}"

            search_col, filter_col, view_col = st.columns([2, 1, 1])

            with search_col:
                search_query = st.text_input(
                    "Search report content",
                    placeholder="Search insights, recommendations, findings...",
                    key=f"report_search{suffix}"
                )

            with filter_col:
                section_filter = st.selectbox(
                    "Filter by section",
                    options=["All Sections", "Executive Summary", "Insights", "Recommendations", "Technical Details", "Metrics"],
                    key=f"section_filter{suffix}"
                )

            with view_col:
                view_mode = st.selectbox(
                    "View mode",
                    options=["Detailed", "Compact", "Overview"],
                    key=f"view_mode{suffix}"
                )

        return {
            "search_query": search_query,
            "section_filter": section_filter,
            "view_mode": view_mode,
            "content_container": content_container
        }

    def search_content(self, analysis_result, query: str) -> List[Dict]:
        """Search through analysis content"""

        if not query:
            return []

        results = []
        query_lower = query.lower()

        # Search executive summary
        if hasattr(analysis_result, 'executive_summary') and analysis_result.executive_summary:
            if query_lower in analysis_result.executive_summary.lower():
                results.append({
                    "section": "Executive Summary",
                    "content": analysis_result.executive_summary,
                    "match_type": "summary",
                    "relevance": self._calculate_relevance(analysis_result.executive_summary, query)
                })

        # Search insights
        if hasattr(analysis_result, 'insights') and analysis_result.insights:
            insights = analysis_result.insights

            # Search strengths
            if hasattr(insights, 'strengths') and insights.strengths:
                for strength in insights.strengths:
                    if query_lower in strength.lower():
                        results.append({
                            "section": "Strengths",
                            "content": strength,
                            "match_type": "strength",
                            "relevance": self._calculate_relevance(strength, query)
                        })

            # Search weaknesses
            if hasattr(insights, 'weaknesses') and insights.weaknesses:
                for weakness in insights.weaknesses:
                    if query_lower in weakness.lower():
                        results.append({
                            "section": "Areas for Improvement",
                            "content": weakness,
                            "match_type": "weakness",
                            "relevance": self._calculate_relevance(weakness, query)
                        })

            # Search opportunities
            if hasattr(insights, 'opportunities') and insights.opportunities:
                for opportunity in insights.opportunities:
                    if query_lower in opportunity.lower():
                        results.append({
                            "section": "Opportunities",
                            "content": opportunity,
                            "match_type": "opportunity",
                            "relevance": self._calculate_relevance(opportunity, query)
                        })

            # Search recommendations
            if hasattr(insights, 'recommendations') and insights.recommendations:
                for recommendation in insights.recommendations:
                    if query_lower in recommendation.lower():
                        results.append({
                            "section": "Recommendations",
                            "content": recommendation,
                            "match_type": "recommendation",
                            "relevance": self._calculate_relevance(recommendation, query)
                        })

        # Sort by relevance
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results

    def _calculate_relevance(self, content: str, query: str) -> float:
        """Calculate relevance score for search results"""

        content_lower = content.lower()
        query_lower = query.lower()

        # Exact match gets highest score
        if query_lower == content_lower:
            return 1.0

        # Word boundary matches
        words = query_lower.split()
        word_matches = sum(1 for word in words if word in content_lower)
        word_score = word_matches / len(words) if words else 0

        # Substring matches
        if query_lower in content_lower:
            substring_score = len(query_lower) / len(content_lower)
        else:
            substring_score = 0

        # Combined score
        return (word_score * 0.7) + (substring_score * 0.3)

    def render_search_results(self, results: List[Dict], query: str):
        """Render search results"""

        if not results:
            st.info(f"No results found for '{query}'")
            return

        st.markdown(f"**Found {len(results)} results for '{query}':**")

        for i, result in enumerate(results):
            with st.expander(f"{result['section']}: {result['content'][:100]}...", expanded=i < 3):

                # Highlight search terms
                highlighted_content = self._highlight_search_terms(result['content'], query)
                st.markdown(highlighted_content, unsafe_allow_html=True)

                # Metadata
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"Section: {result['section']}")
                with col2:
                    st.caption(f"Relevance: {result['relevance']:.2f}")

    def _highlight_search_terms(self, content: str, query: str) -> str:
        """Highlight search terms in content"""

        words = query.lower().split()
        highlighted_content = content

        for word in words:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            highlighted_content = pattern.sub(
                f'<mark style="background-color: #ffeb3b; padding: 2px;">{word}</mark>',
                highlighted_content
            )

        return highlighted_content

    def create_report_outline(self, analysis_result) -> List[Dict]:
        """Create navigable report outline"""

        outline = []

        # Executive Summary
        if hasattr(analysis_result, 'executive_summary') and analysis_result.executive_summary:
            outline.append({
                "section": "Executive Summary",
                "subsections": [],
                "content_preview": analysis_result.executive_summary[:100] + "..."
            })

        # Metrics
        if hasattr(analysis_result, 'metrics') and analysis_result.metrics:
            # Helper function to safely get metric values
            def get_metric_value(metrics, key, default=0):
                if isinstance(metrics, dict):
                    return metrics.get(key, default)
                else:
                    return getattr(metrics, key, default)

            overall_score = get_metric_value(analysis_result.metrics, 'overall_score')
            outline.append({
                "section": "Performance Metrics",
                "subsections": ["Overall Score", "Content Quality", "SEO Score", "UX Score"],
                "content_preview": f"Overall Score: {overall_score:.1f}/10"
            })

        # Insights
        if hasattr(analysis_result, 'insights') and analysis_result.insights:
            # Helper function to safely get insight values
            def get_insight_value(insights, key, default=None):
                if isinstance(insights, dict):
                    return insights.get(key, default if default is not None else [])
                else:
                    return getattr(insights, key, default if default is not None else [])

            insights_subsections = []
            strengths = get_insight_value(analysis_result.insights, 'strengths')
            if strengths:
                insights_subsections.append(f"Strengths ({len(strengths)})")
            weaknesses = get_insight_value(analysis_result.insights, 'weaknesses')
            if weaknesses:
                insights_subsections.append(f"Areas for Improvement ({len(weaknesses)})")
            opportunities = get_insight_value(analysis_result.insights, 'opportunities')
            if opportunities:
                insights_subsections.append(f"Opportunities ({len(opportunities)})")

            outline.append({
                "section": "Detailed Insights",
                "subsections": insights_subsections,
                "content_preview": "Comprehensive analysis findings and observations"
            })

        # Recommendations
        if hasattr(analysis_result, 'insights') and analysis_result.insights and \
           hasattr(analysis_result.insights, 'recommendations') and analysis_result.insights.recommendations:
            outline.append({
                "section": "Recommendations",
                "subsections": [f"Action Item {i+1}" for i in range(len(analysis_result.insights.recommendations))],
                "content_preview": f"{len(analysis_result.insights.recommendations)} actionable recommendations"
            })

        # Technical Details
        outline.append({
            "section": "Technical Details",
            "subsections": ["Analysis Metadata", "Processing Information", "Cost Breakdown"],
            "content_preview": f"Analysis ID: {analysis_result.analysis_id[:8]}..."
        })

        return outline

    def render_navigation_sidebar(self, outline: List[Dict]):
        """Render navigation sidebar"""

        st.sidebar.markdown("---")
        st.sidebar.markdown("**Report Navigation**")

        for section in outline:
            with st.sidebar.expander(section["section"], expanded=False):
                st.caption(section["content_preview"])

                if section["subsections"]:
                    for subsection in section["subsections"]:
                        # Create unique key for navigation buttons
                        unique_nav_key = f"nav_{section['section']}_{subsection}_{hash(subsection) % 10000}"
                        if st.button(f"→ {subsection}", key=unique_nav_key):
                            st.session_state.scroll_to_section = section["section"]
                            st.rerun()

    def create_export_options(self, analysis_result):
        """Create export options for analysis results"""
        st.subheader("Export Report")

        export_format = st.selectbox(
            "Choose export format",
            options=["PDF", "CSV", "JSON"],
            key="export_format_selector"
        )

        if st.button("Generate Export", key="generate_export_btn"):
            if export_format == "PDF":
                self._export_to_pdf(analysis_result)
            elif export_format == "CSV":
                self._export_to_csv(analysis_result)
            elif export_format == "JSON":
                self._export_to_json(analysis_result)

    def _export_to_pdf(self, analysis_result):
        """Export analysis results to PDF"""
        try:
            import io
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            from datetime import datetime

            # Create a BytesIO buffer
            buffer = io.BytesIO()

            # Create PDF document
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []

            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                textColor=colors.HexColor('#1f77b4')
            )
            story.append(Paragraph("Web Content Analysis Report", title_style))
            story.append(Spacer(1, 20))

            # Basic Information
            info_style = styles['Heading2']
            story.append(Paragraph("Analysis Information", info_style))

            info_data = [
                ["URL:", getattr(analysis_result, 'url', 'N/A')],
                ["Analysis Date:", getattr(analysis_result, 'created_at', datetime.now()).strftime("%Y-%m-%d %H:%M:%S")],
                ["Analysis Type:", getattr(analysis_result, 'analysis_type', 'Standard')],
                ["Status:", getattr(analysis_result, 'status', 'Unknown')]
            ]

            info_table = Table(info_data, colWidths=[2*inch, 4*inch])
            info_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.grey),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('BACKGROUND', (1, 0), (1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(info_table)
            story.append(Spacer(1, 20))

            # Metrics Section
            if hasattr(analysis_result, 'metrics') and analysis_result.metrics:
                story.append(Paragraph("Performance Metrics", info_style))

                def get_metric_value(metrics, key, default=0):
                    if isinstance(metrics, dict):
                        return metrics.get(key, default)
                    return getattr(metrics, key, default)

                metrics_data = [
                    ["Metric", "Score", "Rating"],
                    ["Overall Score", f"{get_metric_value(analysis_result.metrics, 'overall_score'):.1f}/10", self._get_rating(get_metric_value(analysis_result.metrics, 'overall_score'))],
                    ["Content Quality", f"{get_metric_value(analysis_result.metrics, 'content_quality_score'):.1f}/10", self._get_rating(get_metric_value(analysis_result.metrics, 'content_quality_score'))],
                    ["SEO Score", f"{get_metric_value(analysis_result.metrics, 'seo_score'):.1f}/10", self._get_rating(get_metric_value(analysis_result.metrics, 'seo_score'))],
                    ["UX Score", f"{get_metric_value(analysis_result.metrics, 'ux_score'):.1f}/10", self._get_rating(get_metric_value(analysis_result.metrics, 'ux_score'))],
                    ["Readability", f"{get_metric_value(analysis_result.metrics, 'readability_score'):.1f}/10", self._get_rating(get_metric_value(analysis_result.metrics, 'readability_score'))],
                    ["Engagement", f"{get_metric_value(analysis_result.metrics, 'engagement_score'):.1f}/10", self._get_rating(get_metric_value(analysis_result.metrics, 'engagement_score'))]
                ]

                metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
                metrics_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(metrics_table)
                story.append(Spacer(1, 20))

            # Insights Section
            if hasattr(analysis_result, 'insights') and analysis_result.insights:
                story.append(Paragraph("Key Insights", info_style))

                insights = analysis_result.insights
                if isinstance(insights, dict):
                    # Handle insights as dictionary
                    for section, items in insights.items():
                        if items and isinstance(items, list):
                            story.append(Paragraph(f"{section.replace('_', ' ').title()}:", styles['Heading3']))
                            for item in items[:5]:  # Limit to 5 items per section
                                story.append(Paragraph(f"• {item}", styles['Normal']))
                            story.append(Spacer(1, 10))
                else:
                    # Handle insights as object
                    for attr in ['strengths', 'weaknesses', 'opportunities', 'recommendations', 'key_findings']:
                        if hasattr(insights, attr):
                            items = getattr(insights, attr)
                            if items:
                                story.append(Paragraph(f"{attr.replace('_', ' ').title()}:", styles['Heading3']))
                                for item in items[:5]:
                                    story.append(Paragraph(f"• {item}", styles['Normal']))
                                story.append(Spacer(1, 10))

            # Executive Summary
            if hasattr(analysis_result, 'executive_summary') and analysis_result.executive_summary:
                story.append(Paragraph("Executive Summary", info_style))
                story.append(Paragraph(analysis_result.executive_summary, styles['Normal']))
                story.append(Spacer(1, 20))

            # Footer
            story.append(Spacer(1, 30))
            footer_style = ParagraphStyle(
                'Footer',
                parent=styles['Normal'],
                fontSize=8,
                textColor=colors.grey
            )
            story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Web Content Analyzer", footer_style))

            # Build PDF
            doc.build(story)

            # Get PDF data
            pdf_data = buffer.getvalue()
            buffer.close()

            # Create download button
            filename = f"analysis_report_{getattr(analysis_result, 'analysis_id', 'report')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_data,
                file_name=filename,
                mime="application/pdf",
                key="download_pdf_btn"
            )

            st.success("PDF report generated successfully!")

        except ImportError:
            st.error("PDF export requires reportlab library. Install with: pip install reportlab")
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")

    def _export_to_csv(self, analysis_result):
        """Export analysis results to CSV"""
        try:
            import pandas as pd
            import io
            from datetime import datetime

            # Prepare data for CSV export
            data = {
                'Metric': [],
                'Value': [],
                'Category': []
            }

            # Basic information
            basic_info = {
                'URL': getattr(analysis_result, 'url', 'N/A'),
                'Analysis Date': getattr(analysis_result, 'created_at', datetime.now()).strftime("%Y-%m-%d %H:%M:%S"),
                'Analysis Type': getattr(analysis_result, 'analysis_type', 'Standard'),
                'Status': getattr(analysis_result, 'status', 'Unknown')
            }

            for key, value in basic_info.items():
                data['Metric'].append(key)
                data['Value'].append(value)
                data['Category'].append('Basic Information')

            # Metrics
            if hasattr(analysis_result, 'metrics') and analysis_result.metrics:
                def get_metric_value(metrics, key, default=0):
                    if isinstance(metrics, dict):
                        return metrics.get(key, default)
                    return getattr(metrics, key, default)

                metrics_info = {
                    'Overall Score': get_metric_value(analysis_result.metrics, 'overall_score'),
                    'Content Quality Score': get_metric_value(analysis_result.metrics, 'content_quality_score'),
                    'SEO Score': get_metric_value(analysis_result.metrics, 'seo_score'),
                    'UX Score': get_metric_value(analysis_result.metrics, 'ux_score'),
                    'Readability Score': get_metric_value(analysis_result.metrics, 'readability_score'),
                    'Engagement Score': get_metric_value(analysis_result.metrics, 'engagement_score')
                }

                for key, value in metrics_info.items():
                    data['Metric'].append(key)
                    data['Value'].append(f"{value:.2f}" if isinstance(value, (int, float)) else str(value))
                    data['Category'].append('Performance Metrics')

            # Insights
            if hasattr(analysis_result, 'insights') and analysis_result.insights:
                insights = analysis_result.insights

                if isinstance(insights, dict):
                    for section, items in insights.items():
                        if items and isinstance(items, list):
                            for i, item in enumerate(items[:10]):  # Limit to 10 items per section
                                data['Metric'].append(f"{section.replace('_', ' ').title()} {i+1}")
                                data['Value'].append(str(item))
                                data['Category'].append('Insights')
                else:
                    for attr in ['strengths', 'weaknesses', 'opportunities', 'recommendations', 'key_findings']:
                        if hasattr(insights, attr):
                            items = getattr(insights, attr)
                            if items:
                                for i, item in enumerate(items[:10]):
                                    data['Metric'].append(f"{attr.replace('_', ' ').title()} {i+1}")
                                    data['Value'].append(str(item))
                                    data['Category'].append('Insights')

            # Create DataFrame
            df = pd.DataFrame(data)

            # Convert to CSV
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            # Create download button
            filename = (
                f"analysis_data_{getattr(analysis_result, 'analysis_id', 'report')}"
                f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )

            st.download_button(
                label="📊 Download CSV Data",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                key="download_csv_btn"
            )

            st.success("CSV data generated successfully!")

            # Show preview
            st.subheader("Data Preview")
            st.dataframe(df.head(20), use_container_width=True)

        except Exception as e:
            st.error(f"Error generating CSV: {str(e)}")

    def _export_to_json(self, analysis_result):
        """Export analysis results to JSON"""
        try:
            import json
            import io
            from datetime import datetime

            # Convert analysis result to dictionary
            export_data = {
                "metadata": {
                    "export_date": datetime.now().isoformat(),
                    "generator": "Web Content Analyzer",
                    "version": "2.4"
                },
                "analysis": {
                    "id": getattr(analysis_result, 'analysis_id', 'unknown'),
                    "url": getattr(analysis_result, 'url', 'N/A'),
                    "created_at": getattr(analysis_result, 'created_at', datetime.now()).isoformat(),
                    "analysis_type": getattr(analysis_result, 'analysis_type', 'Standard'),
                    "status": getattr(analysis_result, 'status', 'Unknown')
                }
            }

            # Add metrics
            if hasattr(analysis_result, 'metrics') and analysis_result.metrics:
                def get_metric_value(metrics, key, default=0):
                    if isinstance(metrics, dict):
                        return metrics.get(key, default)
                    return getattr(metrics, key, default)

                export_data["metrics"] = {
                    "overall_score": get_metric_value(analysis_result.metrics, 'overall_score'),
                    "content_quality_score": get_metric_value(analysis_result.metrics, 'content_quality_score'),
                    "seo_score": get_metric_value(analysis_result.metrics, 'seo_score'),
                    "ux_score": get_metric_value(analysis_result.metrics, 'ux_score'),
                    "readability_score": get_metric_value(analysis_result.metrics, 'readability_score'),
                    "engagement_score": get_metric_value(analysis_result.metrics, 'engagement_score')
                }

            # Add insights
            if hasattr(analysis_result, 'insights') and analysis_result.insights:
                insights = analysis_result.insights
                if isinstance(insights, dict):
                    export_data["insights"] = insights
                else:
                    export_data["insights"] = {}
                    for attr in ['strengths', 'weaknesses', 'opportunities', 'recommendations', 'key_findings']:
                        if hasattr(insights, attr):
                            export_data["insights"][attr] = getattr(insights, attr)

            # Add executive summary
            if hasattr(analysis_result, 'executive_summary') and analysis_result.executive_summary:
                export_data["executive_summary"] = analysis_result.executive_summary

            # Convert to JSON
            json_data = json.dumps(export_data, indent=2, default=str)

            # Create download button
            filename = (
                f"analysis_data_{getattr(analysis_result, 'analysis_id', 'report')}"
                f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

            st.download_button(
                label="📄 Download JSON Data",
                data=json_data,
                file_name=filename,
                mime="application/json",
                key="download_json_btn"
            )

            st.success("JSON data generated successfully!")

            # Show preview
            st.subheader("JSON Preview")
            st.code(json_data[:1000] + "..." if len(json_data) > 1000 else json_data, language="json")

        except Exception as e:
            st.error(f"Error generating JSON: {str(e)}")

    def _get_rating(self, score):
        """Convert numeric score to rating"""
        if score >= 9:
            return "Excellent"
        elif score >= 7:
            return "Good"
        elif score >= 5:
            return "Average"
        elif score >= 3:
            return "Poor"
        else:
            return "Very Poor"


class ReportComparison:
    """Compare multiple analysis reports"""

    def create_comparison_interface(self, analysis_results: List):
        """Create interface for comparing multiple reports"""

        if len(analysis_results) < 2:
            st.info("Need at least 2 analyses for comparison")
            return

        st.subheader("Report Comparison")

        # Select reports to compare
        col1, col2 = st.columns(2)

        with col1:
            report1_idx = st.selectbox(
                "Select first report",
                options=range(len(analysis_results)),
                format_func=lambda x: f"{analysis_results[x].url} ({analysis_results[x].created_at.strftime('%m/%d/%Y')})",
                key="compare_report1"
            )

        with col2:
            report2_idx = st.selectbox(
                "Select second report",
                options=range(len(analysis_results)),
                format_func=lambda x: f"{analysis_results[x].url} ({analysis_results[x].created_at.strftime('%m/%d/%Y')})",
                key="compare_report2"
            )

        if report1_idx == report2_idx:
            st.warning("Please select different reports for comparison")
            return

        # Render comparison
        self.render_side_by_side_comparison(
            analysis_results[report1_idx],
            analysis_results[report2_idx]
        )

    def render_side_by_side_comparison(self, report1, report2):
        """Render side-by-side comparison"""

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Report A: {report1.url}**")
            self.render_compact_report(report1)

        with col2:
            st.markdown(f"**Report B: {report2.url}**")
            self.render_compact_report(report2)

        # Comparison metrics
        st.markdown("---")
        st.subheader("Comparison Summary")

        if hasattr(report1, 'metrics') and hasattr(report2, 'metrics') and \
           report1.metrics and report2.metrics:

            # Helper to safely get metric values
            def m(obj, key, default=0):
                if not obj:
                    return default
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return getattr(obj, key, default)

            # Helper to determine max score dynamically
            def get_max_score(metrics_obj):
                """Determine the maximum possible score from the metrics object or system default"""
                # Method 1: Check if max_score is stored in the metrics
                if isinstance(metrics_obj, dict):
                    if 'max_score' in metrics_obj:
                        return metrics_obj['max_score']
                    if '_metadata' in metrics_obj and 'max_score' in metrics_obj['_metadata']:
                        return metrics_obj['_metadata']['max_score']
                else:
                    if hasattr(metrics_obj, 'max_score'):
                        return metrics_obj.max_score
                    if hasattr(metrics_obj, '_metadata') and hasattr(metrics_obj._metadata, 'max_score'):
                        return metrics_obj._metadata.max_score

                # Method 2: Infer from the highest score in the data (assuming scores are normalized)
                all_scores = []
                for key in ['overall_score', 'content_quality_score', 'seo_score', 'ux_score', 'readability_score', 'engagement_score']:
                    score = m(metrics_obj, key)
                    if score is not None and score > 0:
                        all_scores.append(score)

                if all_scores:
                    max_observed = max(all_scores)
                    # Common score ranges: if max observed is around 1.0, it's probably 0-1 scale
                    if max_observed <= 1.0:
                        return 1.0
                    # If max observed is around 5.0, it's probably 0-5 scale
                    elif max_observed <= 5.0:
                        return 5.0
                    # If max observed is around 10.0, it's probably 0-10 scale
                    elif max_observed <= 10.0:
                        return 10.0
                    # If max observed is around 100.0, it's probably 0-100 scale
                    elif max_observed <= 100.0:
                        return 100.0

                # Method 3: Check application configuration (if available)
                if hasattr(st.session_state, 'app_config') and 'max_score' in st.session_state.app_config:
                    return st.session_state.app_config['max_score']

                # Default fallback to 10 (current system default)
                return 10.0

            # Get the maximum score (use report1's metrics to determine scale)
            max_score = get_max_score(report1.metrics)

            # Get raw values for calculation
            report1_overall = m(report1.metrics, 'overall_score')
            report1_content = m(report1.metrics, 'content_quality_score')
            report1_seo = m(report1.metrics, 'seo_score')
            report1_ux = m(report1.metrics, 'ux_score')

            report2_overall = m(report2.metrics, 'overall_score')
            report2_content = m(report2.metrics, 'content_quality_score')
            report2_seo = m(report2.metrics, 'seo_score')
            report2_ux = m(report2.metrics, 'ux_score')

            # Format scores with dynamic max score
            def format_score(score, max_val):
                if score is None:
                    return "N/A"
                if max_val == 1.0:
                    return f"{score:.2f}/1.0"
                elif max_val == 5.0:
                    return f"{score:.1f}/5.0"
                elif max_val == 10.0:
                    return f"{score:.1f}/10"
                elif max_val == 100.0:
                    return f"{score:.0f}/100"
                else:
                    return f"{score:.1f}/{max_val:.0f}"

            metrics_comparison = pd.DataFrame({
                'Metric': ['Overall Score', 'Content Quality', 'SEO Score', 'UX Score'],
                'Report A': [
                    format_score(report1_overall, max_score),
                    format_score(report1_content, max_score),
                    format_score(report1_seo, max_score),
                    format_score(report1_ux, max_score)
                ],
                'Report B': [
                    format_score(report2_overall, max_score),
                    format_score(report2_content, max_score),
                    format_score(report2_seo, max_score),
                    format_score(report2_ux, max_score)
                ],
                'Difference': [
                    f"{report2_overall - report1_overall:+.1f}" if report2_overall is not None and report1_overall is not None else "N/A",
                    f"{report2_content - report1_content:+.1f}" if report2_content is not None and report1_content is not None else "N/A",
                    f"{report2_seo - report1_seo:+.1f}" if report2_seo is not None and report1_seo is not None else "N/A",
                    f"{report2_ux - report1_ux:+.1f}" if report2_ux is not None and report1_ux is not None else "N/A"
                ]
            })

            # Add better column based on difference
            differences = [
                (report2_overall - report1_overall) if report2_overall is not None and report1_overall is not None else 0,
                (report2_content - report1_content) if report2_content is not None and report1_content is not None else 0,
                (report2_seo - report1_seo) if report2_seo is not None and report1_seo is not None else 0,
                (report2_ux - report1_ux) if report2_ux is not None and report1_ux is not None else 0
            ]

            metrics_comparison['Better'] = [
                'Report B' if diff > 0 else 'Report A' if diff < 0 else 'Equal'
                for diff in differences
            ]

            st.dataframe(metrics_comparison, use_container_width=True)

            # Display the scoring scale information
            st.caption(f"*Scores are on a scale of 0 to {max_score:.0f}")

    def render_compact_report(self, report):
        """Render compact version of report"""

        # Key metrics
        if hasattr(report, 'metrics') and report.metrics:
            val = None
            if isinstance(report.metrics, dict):
                val = report.metrics.get('overall_score')
            else:
                val = getattr(report.metrics, 'overall_score', None)
            if val is not None:
                st.metric("Overall Score", f"{val:.1f}/10")

        # Executive summary (truncated)
        if hasattr(report, 'executive_summary') and report.executive_summary:
            st.markdown("**Summary:**")
            st.write(report.executive_summary[:200] + "...")

        # Top insights
        if hasattr(report, 'insights') and report.insights:
            if hasattr(report.insights, 'key_findings') and report.insights.key_findings:
                st.markdown("**Key Findings:**")
                for finding in report.insights.key_findings[:3]:
                    st.write(f"• {finding}")
