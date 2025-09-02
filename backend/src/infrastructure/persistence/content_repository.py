"""
Repository for managing scraped content and associated images
"""
import sqlite3
import logging
from typing import List, Optional
from datetime import datetime
from pathlib import Path

from src.domain.models import (
    ScrapedContent, ExtractedImage, AnalysisResult, URLInfo, 
    ContentType, ScrapingStatus
)

logger = logging.getLogger(__name__)

class ContentRepository:
    """Repository for managing scraped content and related data"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        logger.info(f"🔍 ContentRepository initializing with db_path: {db_path}")
        self._ensure_database_exists()
    
    def _ensure_database_exists(self):
        """Ensure database and required tables exist"""
        try:
            logger.info(f"🔍 Attempting to connect to database: {self.db_path}")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create scraped_content table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scraped_content (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL UNIQUE,
                    title TEXT,
                    main_content TEXT,
                    headings TEXT,  -- JSON array of headings
                    links TEXT,     -- JSON array of links
                    meta_description TEXT,
                    meta_keywords TEXT,  -- JSON array of keywords
                    content_type TEXT,
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT
                )
            """)
            
            # Create images table if it doesn't exist (from migration)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scraped_content_id INTEGER NOT NULL,
                    url TEXT NOT NULL,
                    alt_text TEXT,
                    title TEXT,
                    context TEXT,
                    image_type TEXT NOT NULL,
                    image_format TEXT NOT NULL,
                    file_size INTEGER,
                    width INTEGER,
                    height INTEGER,
                    file_path TEXT,
                    thumbnail_path TEXT,
                    is_decorative BOOLEAN DEFAULT FALSE,
                    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (scraped_content_id) REFERENCES scraped_content(id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_images_scraped_content_id 
                ON images(scraped_content_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_scraped_content_url 
                ON scraped_content(url)
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error ensuring database exists: {str(e)}")
            if 'conn' in locals():
                conn.close()
    
    def save_scraped_content(self, content: ScrapedContent) -> Optional[int]:
        """
        Save scraped content to database and return the generated ID
        
        Args:
            content: ScrapedContent object to save
            
        Returns:
            Optional[int]: Database ID of saved content, None if failed
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert lists to JSON strings for storage
            import json
            headings_json = json.dumps(content.headings) if content.headings else "[]"
            links_json = json.dumps(content.links) if content.links else "[]"
            keywords_json = json.dumps(content.meta_keywords) if content.meta_keywords else "[]"
            
            cursor.execute("""
                INSERT OR REPLACE INTO scraped_content (
                    url, title, main_content, headings, links,
                    meta_description, meta_keywords, content_type,
                    scraped_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                content.url_info.url,
                content.title,
                content.main_content,
                headings_json,
                links_json,
                content.meta_description,
                keywords_json,
                content.content_type.value if content.content_type else None,
                content.scraped_at.isoformat(),
                content.status.value if content.status else None
            ))
            
            content_id = cursor.lastrowid
            
            # Save associated images if any
            logger.info(f"🔍 ScrapedContent has {len(content.images) if content.images else 0} images")
            if content.images:
                logger.info(f"📸 First few images: {[img.info.original_url[:50] for img in content.images[:3]]}")
                self._save_images(cursor, content_id, content.images)
            else:
                logger.warning("⚠️ No images found in ScrapedContent to save")
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Saved scraped content for {content.url_info.url} with ID {content_id}")
            return content_id
            
        except Exception as e:
            logger.error(f"❌ Error saving scraped content: {str(e)}")
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            return None
    
    def _save_images(self, cursor: sqlite3.Cursor, content_id: int, images: List[ExtractedImage]):
        """Save images associated with scraped content"""
        logger.info(f"Saving {len(images)} images for content_id {content_id}")
        for image in images:
            try:
                cursor.execute("""
                    INSERT INTO images (
                        scraped_content_id, url, alt_text, title, context,
                        image_type, image_format, file_size, width, height,
                        file_path, thumbnail_path, is_decorative, extracted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    content_id, 
                    image.info.original_url, 
                    image.info.alt_text, 
                    image.info.title,
                    image.info.context, 
                    image.info.image_type.value, 
                    image.info.image_format.value,
                    image.info.file_size, 
                    image.info.width, 
                    image.info.height, 
                    image.local_path,
                    image.thumbnail_path, 
                    image.info.is_decorative, 
                    image.extracted_at.isoformat()
                ))
                logger.debug(f"Saved image: {image.info.original_url}")
            except Exception as e:
                logger.error(f"Error saving image {image.info.original_url}: {e}")
        logger.info(f"✅ Saved {len(images)} images to database")
    
    def get_scraped_content_by_url(self, url: str) -> Optional[ScrapedContent]:
        """Get scraped content by URL"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, url, title, main_content, headings, links,
                       meta_description, meta_keywords, content_type,
                       scraped_at, status
                FROM scraped_content 
                WHERE url = ?
            """, (url,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            # Convert JSON strings back to lists
            import json
            headings = json.loads(row[4]) if row[4] else []
            links = json.loads(row[5]) if row[5] else []
            keywords = json.loads(row[7]) if row[7] else []
            
            # Note: This is a simplified reconstruction - some fields may be missing
            # In a full implementation, you'd want to store all URLInfo and metrics
            return ScrapedContent(
                url_info=URLInfo(url=row[1]),
                title=row[2],
                main_content=row[3],
                headings=headings,
                links=links,
                meta_description=row[6],
                meta_keywords=keywords,
                content_type=ContentType(row[8]) if row[8] else None,
                scraped_at=datetime.fromisoformat(row[9]),
                status=ScrapingStatus(row[10]) if row[10] else None,
                images=[]  # Images would be loaded separately if needed
            )
            
        except Exception as e:
            logger.error(f"Error fetching scraped content for {url}: {str(e)}")
            return None
    
    def get_images_for_content(self, content_id: int) -> List[ExtractedImage]:
        """Get all images for a specific scraped content ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, scraped_content_id, url, alt_text, title, context,
                       image_type, image_format, file_size, width, height,
                       file_path, thumbnail_path, is_decorative, extracted_at
                FROM images 
                WHERE scraped_content_id = ?
                ORDER BY extracted_at DESC
            """, (content_id,))
            
            rows = cursor.fetchall()
            conn.close()
            
            images = []
            for row in rows:
                from src.domain.models import ImageType, ImageFormat
                images.append(ExtractedImage(
                    id=row[0],
                    scraped_content_id=row[1],
                    url=row[2],
                    alt_text=row[3],
                    title=row[4],
                    context=row[5],
                    image_type=ImageType(row[6]),
                    image_format=ImageFormat(row[7]),
                    file_size=row[8],
                    width=row[9],
                    height=row[10],
                    file_path=row[11],
                    thumbnail_path=row[12],
                    is_decorative=bool(row[13]),
                    extracted_at=datetime.fromisoformat(row[14])
                ))
            
            return images
            
        except Exception as e:
            logger.error(f"Error fetching images for content_id {content_id}: {str(e)}")
            return []
