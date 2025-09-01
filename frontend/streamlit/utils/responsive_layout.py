"""
Responsive Layout Utilities for Desktop and Tablet
WBS 2.4: Responsive design components and utilities
"""

import streamlit as st
from typing import Dict, List, Optional, Tuple, Any

class ResponsiveLayout:
    """Responsive layout utilities for different screen sizes"""
    
    def __init__(self):
        self.breakpoints = {
            'mobile': 768,
            'tablet': 1024,
            'desktop': 1200
        }
    
    def inject_responsive_css(self):
        """Inject responsive CSS for better mobile/tablet experience"""
        
        responsive_css = """
        <style>
        /* Responsive Design CSS */
        
        /* Mobile First Approach */
        @media screen and (max-width: 768px) {
            .main .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
                max-width: 100%;
            }
            
            .stButton > button {
                width: 100%;
                margin-bottom: 0.5rem;
            }
            
            .metric-container {
                margin-bottom: 1rem;
            }
            
            /* Stack columns on mobile */
            .stColumns {
                flex-direction: column;
            }
            
            .stColumns > div {
                width: 100% !important;
                margin-bottom: 1rem;
            }
            
            /* Adjust text input for mobile */
            .stTextInput > div > div > input {
                font-size: 16px; /* Prevent zoom on iOS */
            }
            
            /* Sidebar adjustments */
            .css-1d391kg {
                width: 100%;
                margin-left: 0;
            }
        }
        
        /* Tablet Specific */
        @media screen and (min-width: 769px) and (max-width: 1024px) {
            .main .block-container {
                padding-left: 2rem;
                padding-right: 2rem;
                max-width: 95%;
            }
            
            /* Sidebar width on tablet */
            .css-1d391kg {
                width: 300px;
            }
            
            /* Better spacing for tablet */
            .element-container {
                margin-bottom: 1rem;
            }
        }
        
        /* Desktop Optimization */
        @media screen and (min-width: 1025px) {
            .main .block-container {
                max-width: 1200px;
                padding-left: 3rem;
                padding-right: 3rem;
            }
            
            /* Desktop sidebar */
            .css-1d391kg {
                width: 350px;
            }
        }
        
        /* Touch-friendly elements */
        .stButton > button {
            min-height: 44px; /* iOS/Android touch target minimum */
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 500;
        }
        
        .stSelectbox > div > div {
            min-height: 44px;
        }
        
        .stTextInput > div > div > input {
            min-height: 44px;
            padding: 0.75rem;
            border-radius: 8px;
        }
        
        /* Progress bars */
        .stProgress > div > div {
            height: 12px;
            border-radius: 6px;
        }
        
        /* Card-like containers */
        .analysis-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border: 1px solid #e0e0e0;
        }
        
        /* Responsive tables */
        .dataframe {
            font-size: 0.9rem;
            overflow-x: auto;
        }
        
        @media screen and (max-width: 768px) {
            .dataframe {
                font-size: 0.8rem;
            }
            
            .dataframe th,
            .dataframe td {
                padding: 0.5rem 0.25rem;
            }
        }
        
        /* Improved readability */
        .metric-value {
            font-size: 2rem;
            font-weight: 600;
            color: #1f2937;
        }
        
        .metric-label {
            font-size: 0.875rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* Loading states */
        .loading-shimmer {
            background: linear-gradient(90deg, #f3f4f6 25%, #e5e7eb 50%, #f3f4f6 75%);
            background-size: 200% 100%;
            animation: shimmer 2s infinite;
        }
        
        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }
        
        /* Dark mode support */
        @media (prefers-color-scheme: dark) {
            .analysis-card {
                background: #1f2937;
                border-color: #374151;
                color: #f9fafb;
            }
            
            .metric-value {
                color: #f9fafb;
            }
        }
        
        /* Accessibility improvements */
        .stButton > button:focus,
        .stSelectbox > div > div:focus,
        .stTextInput > div > div > input:focus {
            outline: 2px solid #3b82f6;
            outline-offset: 2px;
        }
        
        /* High contrast mode support */
        @media (prefers-contrast: high) {
            .analysis-card {
                border: 2px solid #000;
            }
            
            .stButton > button {
                border: 2px solid #000;
            }
        }
        </style>
        """
        
        st.markdown(responsive_css, unsafe_allow_html=True)
    
    def create_responsive_columns(self, 
                                mobile_layout: List[int], 
                                tablet_layout: List[int], 
                                desktop_layout: List[int]) -> List:
        """Create responsive columns based on screen size"""
        
        # For now, use desktop layout (Streamlit limitation)
        # In a real implementation, you'd use JavaScript to detect screen size
        return st.columns(desktop_layout)
    
    def create_mobile_friendly_metrics(self, metrics_data: Dict[str, Any]):
        """Create mobile-friendly metrics display"""
        
        # Check if we should use mobile layout (simplified approach)
        # In production, you'd use JavaScript or browser detection
        
        # Mobile layout: single column
        if len(metrics_data) <= 2:
            cols = st.columns(len(metrics_data))
        elif len(metrics_data) <= 4:
            cols = st.columns(2)  # 2x2 grid on mobile
        else:
            cols = st.columns(3)  # 3 columns max
        
        for i, (label, value) in enumerate(metrics_data.items()):
            with cols[i % len(cols)]:
                self.create_metric_card(label, value)
    
    def create_metric_card(self, label: str, value: Any, delta: Optional[str] = None):
        """Create a responsive metric card"""
        
        container = st.container()
        with container:
            st.markdown(f"""
            <div class="analysis-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                {f'<div class="metric-delta">{delta}</div>' if delta else ''}
            </div>
            """, unsafe_allow_html=True)
    
    def create_responsive_tabs(self, tab_names: List[str], icons: List[str] = None) -> List:
        """Create responsive tabs with optional icons"""
        
        if icons and len(icons) == len(tab_names):
            tab_labels = [f"{icon} {name}" for icon, name in zip(icons, tab_names)]
        else:
            tab_labels = tab_names
        
        return st.tabs(tab_labels)
    
    def create_adaptive_sidebar(self):
        """Create adaptive sidebar that works on mobile"""
        
        with st.sidebar:
            # Add mobile-friendly styling
            st.markdown("""
            <style>
            .css-1d391kg {
                background-color: #f8f9fa;
            }
            
            @media screen and (max-width: 768px) {
                .css-1d391kg {
                    position: relative;
                    width: 100%;
                    height: auto;
                    transform: none;
                }
            }
            </style>
            """, unsafe_allow_html=True)
    
    def create_loading_skeleton(self, lines: int = 3):
        """Create loading skeleton for better UX"""
        
        skeleton_html = """
        <div class="loading-skeleton">
        """
        
        for i in range(lines):
            width = "100%" if i < lines - 1 else "60%"
            skeleton_html += f"""
            <div class="loading-shimmer" style="
                height: 1rem; 
                width: {width}; 
                margin-bottom: 0.5rem; 
                border-radius: 4px;
            "></div>
            """
        
        skeleton_html += "</div>"
        
        st.markdown(skeleton_html, unsafe_allow_html=True)
    
    def create_responsive_container(self, content_func, container_class: str = "analysis-card"):
        """Create responsive container wrapper"""
        
        st.markdown(f'<div class="{container_class}">', unsafe_allow_html=True)
        content_func()
        st.markdown('</div>', unsafe_allow_html=True)


class SessionStateManager:
    """Enhanced session state management for complex applications"""
    
    def __init__(self):
        self.init_base_state()
    
    def init_base_state(self):
        """Initialize base session state variables"""
        
        defaults = {
            'analysis_results': [],
            'current_analysis': None,
            'analysis_history': [],
            'selected_analyses': [],
            'search_query': '',
            'filters': {},
            'view_mode': 'detailed',
            'current_page': 'Analysis',  # Match radio button option exactly
            'sidebar_collapsed': False,
            'theme_mode': 'light',
            'user_preferences': {
                'auto_save': True,
                'show_progress_details': True,
                'notifications_enabled': True
            }
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def save_user_preferences(self, preferences: Dict):
        """Save user preferences to session state"""
        st.session_state.user_preferences.update(preferences)
    
    def get_user_preference(self, key: str, default=None):
        """Get user preference value"""
        return st.session_state.user_preferences.get(key, default)
    
    def clear_analysis_data(self):
        """Clear analysis-related session data"""
        st.session_state.analysis_results = []
        st.session_state.current_analysis = None
        st.session_state.selected_analyses = []
    
    def backup_session_state(self) -> Dict:
        """Create backup of current session state"""
        
        backup_keys = [
            'analysis_results', 'analysis_history', 'user_preferences'
        ]
        
        backup = {}
        for key in backup_keys:
            if key in st.session_state:
                backup[key] = st.session_state[key]
        
        return backup
    
    def restore_session_state(self, backup: Dict):
        """Restore session state from backup"""
        
        for key, value in backup.items():
            st.session_state[key] = value
    
    def get_session_info(self) -> Dict:
        """Get session information for debugging"""
        
        return {
            'total_analyses': len(st.session_state.get('analysis_results', [])),
            'current_analysis_id': st.session_state.get('current_analysis', {}).get('analysis_id') if st.session_state.get('current_analysis') else None,
            'search_active': bool(st.session_state.get('search_query')),
            'filters_active': bool(st.session_state.get('filters')),
            'selected_count': len(st.session_state.get('selected_analyses', [])),
            'current_page': st.session_state.get('current_page'),
            'theme_mode': st.session_state.get('theme_mode')
        }


# Global instances
responsive_layout = ResponsiveLayout()
session_manager = SessionStateManager()
