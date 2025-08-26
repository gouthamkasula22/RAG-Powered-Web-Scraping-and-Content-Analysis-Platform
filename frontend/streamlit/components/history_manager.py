"""
Enhanced History Manager with Persistent Storage
WBS 2.4: Robust analysis history for 50+ analyses with advanced features
"""

import streamlit as st
import sqlite3
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from pathlib import Path
import os

class AnalysisHistoryManager:
    def _ensure_table_exists(self):
        """Create analysis_history table if it does not exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id TEXT PRIMARY KEY,
                    url TEXT,
                    analysis_type TEXT,
                    status TEXT,
                    overall_score REAL,
                    cost REAL,
                    created_at TEXT,
                    executive_summary TEXT,
                    insights TEXT
                )
            ''')
            conn.commit()
    def bulk_delete(self, selected_ids: list) -> bool:
        """Delete analyses with the given IDs from persistent storage (SQLite) and session state."""
        self.init_database()
        deleted_count = 0
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for analysis_id in selected_ids:
                    cursor.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
                    deleted_count += cursor.rowcount
                conn.commit()
        except Exception as e:
            st.error(f"Failed to delete from database: {e}")
            return False

        # Remove from session_state.analysis_results
        if 'analysis_results' in st.session_state:
            st.session_state.analysis_results = [a for a in st.session_state.analysis_results if a.get('id', getattr(a, 'id', None)) not in selected_ids]

        return deleted_count > 0
    """Enhanced history manager with SQLite persistence and advanced features"""
    
    def __init__(self, db_path: str = "data/analysis_history.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for persistent storage"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    overall_score REAL,
                    content_quality_score REAL,
                    seo_score REAL,
                    ux_score REAL,
                    cost REAL,
                    processing_time REAL,
                    provider_used TEXT,
                    created_at TIMESTAMP NOT NULL,
                    executive_summary TEXT,
                    insights_json TEXT,
                    full_result_json TEXT,
                    tags TEXT,
                    notes TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON analyses(created_at);
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_url ON analyses(url);
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON analyses(status);
            """)
    
    def save_analysis(self, analysis_result) -> bool:
        """Save analysis result to database"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Prepare data
                insights_json = None
                if hasattr(analysis_result, 'insights') and analysis_result.insights:
                    insights_json = json.dumps({
                        'strengths': getattr(analysis_result.insights, 'strengths', []),
                        'weaknesses': getattr(analysis_result.insights, 'weaknesses', []),
                        'opportunities': getattr(analysis_result.insights, 'opportunities', []),
                        'recommendations': getattr(analysis_result.insights, 'recommendations', []),
                        'key_findings': getattr(analysis_result.insights, 'key_findings', [])
                    })
                
                # Convert full result to JSON (simplified version)
                full_result_json = json.dumps({
                    'analysis_id': analysis_result.analysis_id,
                    'url': analysis_result.url,
                    'analysis_type': analysis_result.analysis_type.value if hasattr(analysis_result.analysis_type, 'value') else str(analysis_result.analysis_type),
                    'status': analysis_result.status.value if hasattr(analysis_result.status, 'value') else str(analysis_result.status),
                    'executive_summary': getattr(analysis_result, 'executive_summary', ''),
                    'created_at': analysis_result.created_at.isoformat() if analysis_result.created_at else datetime.now().isoformat()
                })
                
                # Insert or replace
                # Helper to safely extract metric values from either dict or object
                def get_metric(metrics_obj, key):
                    try:
                        if not metrics_obj:
                            return None
                        if isinstance(metrics_obj, dict):
                            return metrics_obj.get(key)
                        return getattr(metrics_obj, key, None)
                    except Exception:
                        return None

                conn.execute("""
                    INSERT OR REPLACE INTO analyses (
                        id, url, analysis_type, status, overall_score, content_quality_score,
                        seo_score, ux_score, cost, processing_time, provider_used, created_at,
                        executive_summary, insights_json, full_result_json, tags, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    analysis_result.analysis_id,
                    analysis_result.url,
                    analysis_result.analysis_type.value if hasattr(analysis_result.analysis_type, 'value') else str(analysis_result.analysis_type),
                    analysis_result.status.value if hasattr(analysis_result.status, 'value') else str(analysis_result.status),
                    get_metric(getattr(analysis_result, 'metrics', None), 'overall_score'),
                    get_metric(getattr(analysis_result, 'metrics', None), 'content_quality_score'),
                    get_metric(getattr(analysis_result, 'metrics', None), 'seo_score'),
                    get_metric(getattr(analysis_result, 'metrics', None), 'ux_score'),
                    getattr(analysis_result, 'cost', 0.0),
                    getattr(analysis_result, 'processing_time', 0.0),
                    getattr(analysis_result, 'provider_used', ''),
                    analysis_result.created_at.isoformat() if analysis_result.created_at else datetime.now().isoformat(),
                    getattr(analysis_result, 'executive_summary', ''),
                    insights_json,
                    full_result_json,
                    '',  # tags - empty for now
                    ''   # notes - empty for now
                ))
                
            return True
            
        except Exception as e:
            st.error(f"Failed to save analysis: {e}")
            return False
    
    def load_analyses(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Load analyses from database with pagination"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM analyses 
                    ORDER BY created_at DESC 
                    LIMIT ? OFFSET ?
                """, (limit, offset))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            st.error(f"Failed to load analyses: {e}")
            return []
    
    def search_analyses(self, query: str, filters: Dict = None) -> List[Dict]:
        """Search analyses with text and filters"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Build search query
                where_conditions = []
                params = []
                
                if query:
                    where_conditions.append("""
                        (url LIKE ? OR executive_summary LIKE ? OR insights_json LIKE ?)
                    """)
                    query_param = f"%{query}%"
                    params.extend([query_param, query_param, query_param])
                
                if filters:
                    if filters.get('status'):
                        where_conditions.append("status = ?")
                        params.append(filters['status'])
                    
                    if filters.get('analysis_type'):
                        where_conditions.append("analysis_type = ?")
                        params.append(filters['analysis_type'])
                    
                    if filters.get('date_from'):
                        where_conditions.append("created_at >= ?")
                        params.append(filters['date_from'])
                    
                    if filters.get('date_to'):
                        where_conditions.append("created_at <= ?")
                        params.append(filters['date_to'])
                    
                    if filters.get('min_score'):
                        where_conditions.append("overall_score >= ?")
                        params.append(filters['min_score'])
                
                where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
                
                query_sql = f"""
                    SELECT * FROM analyses 
                    {where_clause}
                    ORDER BY created_at DESC 
                    LIMIT 100
                """
                
                cursor = conn.execute(query_sql, params)
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            st.error(f"Search failed: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """Get analysis statistics"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = {}
                
                # Total analyses
                cursor = conn.execute("SELECT COUNT(*) FROM analyses")
                stats['total_analyses'] = cursor.fetchone()[0]
                
                # Successful analyses
                cursor = conn.execute("SELECT COUNT(*) FROM analyses WHERE status = 'completed'")
                stats['successful_analyses'] = cursor.fetchone()[0]
                
                # Total cost
                cursor = conn.execute("SELECT SUM(cost) FROM analyses WHERE cost IS NOT NULL")
                result = cursor.fetchone()[0]
                stats['total_cost'] = result if result else 0.0
                
                # Average score
                cursor = conn.execute("SELECT AVG(overall_score) FROM analyses WHERE overall_score IS NOT NULL")
                result = cursor.fetchone()[0]
                stats['average_score'] = result if result else 0.0
                
                # Analyses this week
                week_ago = (datetime.now() - timedelta(days=7)).isoformat()
                cursor = conn.execute("SELECT COUNT(*) FROM analyses WHERE created_at >= ?", (week_ago,))
                stats['analyses_this_week'] = cursor.fetchone()[0]
                
                # Most analyzed domain
                cursor = conn.execute("""
                    SELECT url, COUNT(*) as count FROM analyses 
                    GROUP BY url ORDER BY count DESC LIMIT 1
                """)
                result = cursor.fetchone()
                if result:
                    stats['most_analyzed_url'] = result[0]
                    stats['most_analyzed_count'] = result[1]
                
                return stats
                
        except Exception as e:
            st.error(f"Failed to get statistics: {e}")
            return {}
    
    def render_history_interface(self):
        """Render enhanced history management interface"""
        
        st.header("Analysis History")
        
        # Statistics dashboard
        stats = self.get_statistics()
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Analyses", stats.get('total_analyses', 0))
            with col2:
                success_rate = (stats.get('successful_analyses', 0) / max(stats.get('total_analyses', 1), 1)) * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")
            with col3:
                st.metric("Total Cost", f"${stats.get('total_cost', 0):.3f}")
            with col4:
                st.metric("Avg Score", f"{stats.get('average_score', 0):.1f}/10")
        
        # Search and filter interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_query = st.text_input(
                "Search analyses",
                placeholder="Search by URL, summary, or insights...",
                key="history_search"
            )
        
        with col2:
            search_button = st.button("Search", type="primary")
        
        # Advanced filters
        with st.expander("Advanced Filters", expanded=False):
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                status_filter = st.selectbox(
                    "Status",
                    options=["All", "completed", "failed", "pending"],
                    key="status_filter"
                )
                
                analysis_type_filter = st.selectbox(
                    "Analysis Type",
                    options=["All", "comprehensive", "seo_focused", "ux_focused", "content_quality"],
                    key="type_filter"
                )
            
            with filter_col2:
                date_range = st.date_input(
                    "Date Range",
                    value=[],
                    key="date_filter"
                )
                
                min_score = st.slider(
                    "Minimum Score",
                    min_value=0.0,
                    max_value=10.0,
                    value=0.0,
                    step=0.1,
                    key="score_filter"
                )
            
            with filter_col3:
                apply_filters = st.button("Apply Filters")
                clear_filters = st.button("Clear Filters")
        
        # Load and display analyses
        if search_button or apply_filters or search_query:
            filters = {}
            
            if status_filter != "All":
                filters['status'] = status_filter
            if analysis_type_filter != "All":
                filters['analysis_type'] = analysis_type_filter
            if date_range and len(date_range) == 2:
                filters['date_from'] = date_range[0].isoformat()
                filters['date_to'] = date_range[1].isoformat()
            if min_score > 0:
                filters['min_score'] = min_score
            
            analyses = self.search_analyses(search_query, filters)
        else:
            analyses = self.load_analyses(limit=50)
        
        # Display results
        if analyses:
            self.render_analysis_table(analyses)
        else:
            st.info("No analyses found matching your criteria")
    
    def render_analysis_table(self, analyses: List[Dict]):
        """Render interactive analysis table"""
        
        # Prepare data for display
        display_data = []
        for analysis in analyses:
            display_data.append({
                "Select": False,
                "URL": analysis['url'][:50] + "..." if len(analysis['url']) > 50 else analysis['url'],
                "Type": analysis['analysis_type'].replace('_', ' ').title(),
                "Status": analysis['status'].title(),
                "Score": f"{analysis['overall_score']:.1f}" if analysis['overall_score'] else "N/A",
                "Cost": f"${analysis['cost']:.4f}" if analysis['cost'] else "$0.0000",
                "Date": datetime.fromisoformat(analysis['created_at']).strftime('%m/%d/%Y'),
                "Time": datetime.fromisoformat(analysis['created_at']).strftime('%H:%M'),
                "ID": analysis['id']
            })
        
        if not display_data:
            return
        
        # Create editable dataframe
        df = pd.DataFrame(display_data)
        
        edited_df = st.data_editor(
            df,
            column_config={
                "Select": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select analyses for bulk operations",
                    default=False,
                ),
                "URL": st.column_config.TextColumn(
                    "Website URL",
                    help="The analyzed website URL",
                    max_chars=50,
                ),
                "Score": st.column_config.NumberColumn(
                    "Score",
                    help="Overall analysis score",
                    min_value=0,
                    max_value=10,
                    format="%.1f",
                ),
            },
            disabled=["URL", "Type", "Status", "Score", "Cost", "Date", "Time", "ID"],
            hide_index=True,
            use_container_width=True
        )
        
        # Bulk operations
        selected_analyses = edited_df[edited_df['Select'] == True]['ID'].tolist()
        
        if selected_analyses:
            st.markdown(f"**{len(selected_analyses)} analyses selected**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Export Selected", type="primary"):
                    filepath = self.export_to_csv(selected_analyses)
                    if filepath:
                        st.success(f"Exported to {filepath}")
            
            with col2:
                if st.button("Delete Selected", type="secondary"):
                    if st.session_state.get('confirm_delete'):
                        if self.bulk_delete(selected_analyses):
                            st.success(f"Deleted {len(selected_analyses)} analyses")
                            st.rerun()
                    else:
                        st.session_state.confirm_delete = True
                        st.warning("Click again to confirm deletion")
            
            with col3:
                if st.button("View Details"):
                    # Store selected for detailed view
                    st.session_state.selected_for_detail = selected_analyses
                    st.info("Selected analyses ready for detailed view")
            # Display details for selected analyses
            if st.session_state.get('selected_for_detail'):
                selected_ids = st.session_state.selected_for_detail
                details = [a for a in analyses if a['id'] in selected_ids]
                if details:
                    st.markdown("---")
                    st.subheader("Selected Analysis Details")
                    for detail in details:
                        st.markdown(f"**URL:** {detail['url']}")
                        st.markdown(f"**Type:** {detail['analysis_type'].replace('_', ' ').title()}")
                        st.markdown(f"**Status:** {detail['status'].title()}")
                        st.markdown(f"**Score:** {detail['overall_score']:.1f}" if detail['overall_score'] else "N/A")
                        st.markdown(f"**Cost:** ${detail['cost']:.4f}" if detail['cost'] else "$0.0000")
                        st.markdown(f"**Date:** {datetime.fromisoformat(detail['created_at']).strftime('%m/%d/%Y')}")
                        st.markdown(f"**Time:** {datetime.fromisoformat(detail['created_at']).strftime('%H:%M')}")
                        if 'executive_summary' in detail and detail['executive_summary']:
                            st.markdown(f"**Executive Summary:** {detail['executive_summary']}")
                        if 'insights' in detail and detail['insights']:
                            st.markdown("**Insights:**")
                            for k, v in detail['insights'].items():
                                st.markdown(f"- **{k.replace('_', ' ').title()}:** {v}")
                        st.markdown("---")


# Global history manager instance
history_manager = AnalysisHistoryManager()
