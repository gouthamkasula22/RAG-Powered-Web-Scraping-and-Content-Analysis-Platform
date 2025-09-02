"""
Image Gallery Component for displaying extracted images from web content analysis
"""
import streamlit as st
import requests
from typing import List, Dict, Any, Optional
import logging
from PIL import Image
import io
import base64

logger = logging.getLogger(__name__)

class ImageGallery:
    """Component for displaying and managing extracted images"""
    
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url.rstrip('/')
    
    def render_image_gallery(self, content_id: int, title: str = "Extracted Images"):
        """
        Render image gallery for a specific scraped content
        
        Args:
            content_id: ID of the scraped content
            title: Title for the image gallery section
        """
        try:
            # Fetch images from backend
            images = self._fetch_images(content_id)
            
            if not images:
                st.info("📷 No images found for this content")
                return
            
            st.subheader(f"📷 {title} ({len(images)} images)")
            
            # Image type filter
            col1, col2, col3 = st.columns(3)
            with col1:
                show_content = st.checkbox("Content Images", value=True, key=f"show_content_{content_id}")
            with col2:
                show_decorative = st.checkbox("Decorative Images", value=False, key=f"show_decorative_{content_id}")
            with col3:
                image_type_filter = st.selectbox(
                    "Filter by Type",
                    options=["All", "Photo", "Illustration", "Icon", "Logo", "Chart", "Screenshot", "Avatar"],
                    key=f"image_type_{content_id}"
                )
            
            # Apply filters
            filtered_images = self._filter_images(images, show_content, show_decorative, image_type_filter)
            
            if not filtered_images:
                st.info("No images match the current filters")
                return
            
            # Display images in grid
            self._render_image_grid(filtered_images, content_id)
            
            # Display image statistics
            self._render_image_stats(images, content_id)
            
        except Exception as e:
            logger.error(f"Error rendering image gallery: {str(e)}")
            st.error(f"Failed to load images: {str(e)}")
    
    def _fetch_images(self, content_id: int) -> List[Dict[str, Any]]:
        """Fetch images from backend API"""
        try:
            response = requests.get(
                f"{self.backend_url}/api/images/content/{content_id}",
                timeout=20  # Increased timeout for better reliability
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching images: {str(e)}")
            return []
    
    def _filter_images(self, images: List[Dict], show_content: bool, show_decorative: bool, image_type_filter: str) -> List[Dict]:
        """Apply filters to image list"""
        filtered = []
        
        for image in images:
            # Filter by decorative status
            if image.get('is_decorative', False):
                if not show_decorative:
                    continue
            else:
                if not show_content:
                    continue
            
            # Filter by type
            if image_type_filter != "All" and image.get('image_type', '').upper() != image_type_filter.upper():
                continue
            
            filtered.append(image)
        
        return filtered
    
    def _render_image_grid(self, images: List[Dict], content_id: int):
        """Render images in a responsive grid layout"""
        # Group images in rows of 3
        for i in range(0, len(images), 3):
            row_images = images[i:i+3]
            cols = st.columns(len(row_images))
            
            for col, image in zip(cols, row_images):
                with col:
                    self._render_single_image(image, content_id)
    
    def _render_single_image(self, image: Dict[str, Any], content_id: int):
        """Render a single image with metadata"""
        try:
            # Create container for image
            with st.container():
                # Display image thumbnail
                if image.get('thumbnail_url') and image.get('file_path'):
                    # Image was downloaded and thumbnail is available
                    thumbnail_url = f"{self.backend_url}{image['thumbnail_url']}"
                    
                    try:
                        response = requests.get(thumbnail_url, timeout=15)  # Increased for lazy generation
                        response.raise_for_status()
                        
                        # Display image
                        img = Image.open(io.BytesIO(response.content))
                        
                        # Convert to RGB if necessary
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        st.image(img, use_container_width=True, caption=f"Image {image.get('id')}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to load thumbnail: {str(e)}")
                        st.error(f"🖼️ Failed to load image: {str(e)}")
                elif image.get('file_path'):
                    # Image was downloaded but thumbnail might be generating
                    st.info("🔄 Thumbnail generating... Please refresh if needed")
                else:
                    # Image was found but not downloaded
                    st.info(f"🔗 Image URL found: {image.get('url', 'Unknown')[:50]}...")
                    st.caption("💡 Enable 'Download Images' to see thumbnails")
                
                # Image metadata
                st.caption(f"**Type:** {image.get('image_type', 'Unknown')}")
                
                if image.get('alt_text'):
                    st.caption(f"**Alt:** {image['alt_text'][:50]}...")
                
                if image.get('width') and image.get('height'):
                    st.caption(f"**Size:** {image['width']}×{image['height']}")
                
                if image.get('file_size'):
                    file_size = image.get('file_size', 0)
                    if file_size and file_size > 0:
                        size_kb = file_size / 1024
                        st.caption(f"**File:** {size_kb:.1f} KB")
                
                # Action buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔍 View", key=f"view_{image['id']}_{content_id}"):
                        self._show_image_modal(image)
                
                with col2:
                    if st.button("📥 Download", key=f"download_{image['id']}_{content_id}"):
                        self._download_image(image)
                
                # Decorative indicator
                if image.get('is_decorative', False):
                    st.caption("🎨 *Decorative*")
                
        except Exception as e:
            logger.error(f"Error rendering image {image.get('id')}: {str(e)}")
            st.error("Failed to render image")
    
    def _show_image_modal(self, image: Dict[str, Any]):
        """Show detailed image information in modal"""
        # Use session state to track modal
        modal_key = f"modal_{image['id']}"
        
        if modal_key not in st.session_state:
            st.session_state[modal_key] = False
        
        if st.session_state[modal_key]:
            with st.expander(f"Image Details - {image.get('title', 'Untitled')}", expanded=True):
                
                # Display full image if available
                if image.get('download_url'):
                    download_url = f"{self.backend_url}{image['download_url']}"
                    try:
                        response = requests.get(download_url, timeout=10)
                        response.raise_for_status()
                        img = Image.open(io.BytesIO(response.content))
                        st.image(img, caption=image.get('alt_text', ''), use_container_width=True)
                    except Exception as e:
                        st.error(f"Failed to load full image: {str(e)}")
                
                # Detailed metadata
                file_size = image.get('file_size', 0) or 0
                st.json({
                    "URL": image.get('url', ''),
                    "Alt Text": image.get('alt_text', ''),
                    "Title": image.get('title', ''),
                    "Context": image.get('context', ''),
                    "Type": image.get('image_type', ''),
                    "Format": image.get('image_format', ''),
                    "Dimensions": f"{image.get('width', 0)}×{image.get('height', 0)}",
                    "File Size": f"{(file_size / 1024):.1f} KB",
                    "Is Decorative": image.get('is_decorative', False),
                    "Extracted At": image.get('extracted_at', '')
                })
                
                if st.button("Close", key=f"close_{image['id']}"):
                    st.session_state[modal_key] = False
                    st.rerun()
        
        # Set modal state
        st.session_state[modal_key] = True
    
    def _download_image(self, image: Dict[str, Any]):
        """Handle image download"""
        try:
            if image.get('download_url'):
                download_url = f"{self.backend_url}{image['download_url']}"
                st.markdown(f"[📥 Download Full Image]({download_url})")
            else:
                st.warning("Download not available for this image")
        except Exception as e:
            logger.error(f"Error setting up download: {str(e)}")
            st.error("Failed to set up download")
    
    def _render_image_stats(self, images: List[Dict], content_id: int):
        """Render image statistics"""
        try:
            if not images:
                st.info("No images to show statistics for")
                return
            
            with st.expander("📊 Image Statistics", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                
                # Basic counts
                total_images = len(images)
                content_images = len([img for img in images if not img.get('is_decorative', False)])
                decorative_images = len([img for img in images if img.get('is_decorative', False)])
                total_size = sum(img.get('file_size', 0) for img in images if img.get('file_size'))
                
                with col1:
                    st.metric("Total Images", total_images)
                
                with col2:
                    st.metric("Content Images", content_images)
                
                with col3:
                    st.metric("Decorative Images", decorative_images)
                
                with col4:
                    if total_size > 0:
                        st.metric("Total Size", f"{(total_size / 1024):.1f} KB")
                    else:
                        st.metric("Total Size", "Unknown")
                
                # Type breakdown
                type_counts = {}
                format_counts = {}
                
                for image in images:
                    img_type = image.get('image_type', 'Unknown')
                    img_format = image.get('image_format', 'Unknown')
                    
                    type_counts[img_type] = type_counts.get(img_type, 0) + 1
                    format_counts[img_format] = format_counts.get(img_format, 0) + 1
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**By Type:**")
                    if type_counts:
                        for img_type, count in sorted(type_counts.items()):
                            st.write(f"• {img_type}: {count}")
                    else:
                        st.write("No type data available")
                
                with col2:
                    st.write("**By Format:**")
                    if format_counts:
                        for img_format, count in sorted(format_counts.items()):
                            st.write(f"• {img_format}: {count}")
                    else:
                        st.write("No format data available")
        
        except Exception as e:
            logger.error(f"Error rendering image stats: {str(e)}")
            st.error(f"Failed to render image statistics: {str(e)}")

def render_image_gallery(content_id: int, title: str = "Extracted Images"):
    """
    Convenience function to render image gallery
    
    Args:
        content_id: ID of the scraped content
        title: Title for the image gallery section
    """
    gallery = ImageGallery()
    gallery.render_image_gallery(content_id, title)
