#!/usr/bin/env python3
"""
Test Trash Icon UI - Visual demonstration
"""
import sys
import os

# Add the project root to Python path (adjust as needed)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import streamlit after setting up paths
import streamlit as st

def test_trash_icon_ui():
    """Demonstrate the new trash icon UI"""
    
    st.title("🗑️ Trash Icon UI Demo")
    
    st.markdown("""
    <style>
    /* Trash icon button styling - specific to delete buttons */
    div[data-testid="column"]:last-child .stButton > button {
        background: transparent !important;
        border: none !important;
        color: #dc3545 !important;
        font-size: 18px !important;
        padding: 4px 8px !important;
        min-height: 2rem !important;
        height: 2rem !important;
        width: 2rem !important;
        border-radius: 4px !important;
        transition: all 0.2s !important;
        opacity: 0.7 !important;
    }
    div[data-testid="column"]:last-child .stButton > button:hover {
        background: rgba(220, 53, 69, 0.1) !important;
        color: #c82333 !important;
        opacity: 1.0 !important;
    }
    div[data-testid="column"]:last-child .stButton > button:focus {
        box-shadow: none !important;
        outline: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.subheader("Website List with Trash Icons")
    
    # Mock website data
    websites = [
        {"title": "Example Website 1", "url": "https://example1.com", "chunk_count": 15},
        {"title": "Very Long Website Title That Might Wrap", "url": "https://very-long-domain-name-example.com", "chunk_count": 8},
        {"title": "Another Site", "url": "https://another.com", "chunk_count": 23}
    ]
    
    for i, website in enumerate(websites):
        col1, col2 = st.columns([10, 1])
        
        with col1:
            st.markdown(f"""
            <div style="padding: 8px 0;">
                <h4 style="
                    margin: 0 0 0.5rem 0;
                    font-size: 0.875rem;
                    font-weight: 600;
                    color: #2c3e50;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                ">{website['title']}</h4>
                <p style="
                    margin: 0 0 0.5rem 0;
                    font-size: 0.75rem;
                    color: #6c757d;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                ">{website['url']}</p>
                <div style="display: flex; gap: 1rem; font-size: 0.75rem; color: #6c757d;">
                    <span>{website['chunk_count']} chunks</span>
                    <span>2025-08-27</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button("🗑️", key=f"demo_delete_{i}", help="Delete website", 
                       type="secondary", use_container_width=False):
                st.success(f"Would delete: {website['title']}")
                st.balloons()
    
    st.markdown("---")
    st.markdown("**Key Features:**")
    st.markdown("- ✅ **Compact Design**: Trash icon takes minimal space")
    st.markdown("- ✅ **Professional Look**: Clean, no background clutter") 
    st.markdown("- ✅ **Hover Effect**: Subtle highlight on mouse over")
    st.markdown("- ✅ **Tooltip**: Shows 'Delete website' on hover")
    st.markdown("- ✅ **Responsive**: Works well on different screen sizes")

if __name__ == "__main__":
    test_trash_icon_ui()
