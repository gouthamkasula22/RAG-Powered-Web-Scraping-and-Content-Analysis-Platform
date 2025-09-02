"""
FastAPI endpoints for image management and serving
"""
from fastapi import APIRouter, HTTPException, Query, Path as FastAPIPath, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import logging
import os
from pathlib import Path
import sqlite3
from datetime import datetime

from src.domain.models import ExtractedImage, ImageInfo, ImageType, ImageFormat

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/images", tags=["images"])

class ImageRepository:
    """Repository for managing image data"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def get_images_by_content_id(self, content_id: int) -> List[ExtractedImage]:
        """Get all images for a specific scraped content"""
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
                # Create ImageInfo object first
                image_info = ImageInfo(
                    original_url=row[2],
                    alt_text=row[3],
                    title=row[4],
                    width=row[9],
                    height=row[10],
                    file_size=row[8],
                    image_format=ImageFormat(row[7]),
                    image_type=ImageType(row[6]),
                    context=row[5],
                    is_decorative=bool(row[13])
                )
                
                # Create ExtractedImage with ImageInfo and database ID
                images.append(ExtractedImage(
                    info=image_info,
                    local_path=row[11],
                    thumbnail_path=row[12],
                    extracted_at=datetime.fromisoformat(row[14]),
                    id=row[0]  # Add database ID
                ))
            
            return images
            
        except Exception as e:
            logger.error(f"Error fetching images for content_id {content_id}: {str(e)}")
            return []
    
    def get_image_by_id(self, image_id: int) -> Optional[ExtractedImage]:
        """Get a specific image by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, scraped_content_id, url, alt_text, title, context,
                       image_type, image_format, file_size, width, height,
                       file_path, thumbnail_path, is_decorative, extracted_at
                FROM images 
                WHERE id = ?
            """, (image_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            # Create ImageInfo object first
            image_info = ImageInfo(
                original_url=row[2],
                alt_text=row[3],
                title=row[4],
                width=row[9],
                height=row[10],
                file_size=row[8],
                image_format=ImageFormat(row[7]),
                image_type=ImageType(row[6]),
                context=row[5],
                is_decorative=bool(row[13])
            )
            
            # Create ExtractedImage with ImageInfo and database ID
            return ExtractedImage(
                info=image_info,
                local_path=row[11],
                thumbnail_path=row[12],
                extracted_at=datetime.fromisoformat(row[14]),
                id=row[0]  # Add database ID
            )
            
        except Exception as e:
            logger.error(f"Error fetching image {image_id}: {str(e)}")
            return None
    
    def save_images(self, images: List[ExtractedImage]) -> bool:
        """Save extracted images to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for image in images:
                cursor.execute("""
                    INSERT INTO images (
                        scraped_content_id, url, alt_text, title, context,
                        image_type, image_format, file_size, width, height,
                        file_path, thumbnail_path, is_decorative, extracted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    image.scraped_content_id, image.url, image.alt_text, image.title,
                    image.context, image.image_type.value, image.image_format.value,
                    image.file_size, image.width, image.height, image.file_path,
                    image.thumbnail_path, image.is_decorative, image.extracted_at.isoformat()
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Saved {len(images)} images to database")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving images to database: {str(e)}")
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            return False

# Initialize repository (will be dependency injected)
def get_image_repository() -> ImageRepository:
    """Dependency injection for ImageRepository"""
    db_path = os.getenv("DATABASE_PATH", "data/analysis_history.db")
    return ImageRepository(db_path)

# API Endpoints
@router.get("/content/{content_id}", response_model=List[dict])
async def get_images_for_content(
    content_id: int = FastAPIPath(..., description="ID of the scraped content"),
    image_type: Optional[ImageType] = Query(None, description="Filter by image type"),
    include_decorative: bool = Query(True, description="Include decorative images"),
    repo: ImageRepository = Depends(get_image_repository)
):
    """Get all images for a specific scraped content"""
    try:
        images = repo.get_images_by_content_id(content_id)
        
        # Apply filters
        if image_type:
            images = [img for img in images if img.image_type == image_type]
        
        if not include_decorative:
            images = [img for img in images if not img.is_decorative]
        
        # Convert to dict for JSON response
        result = []
        for image in images:
            result.append({
                "id": image.id,  # Use actual database ID
                "scraped_content_id": content_id,  # Use the provided content_id
                "url": image.info.original_url,
                "alt_text": image.info.alt_text,
                "title": image.info.title,
                "context": image.info.context,
                "image_type": image.info.image_type.value,
                "image_format": image.info.image_format.value,
                "file_size": image.info.file_size,
                "width": image.info.width,
                "height": image.info.height,
                "file_path": image.local_path,
                "thumbnail_path": image.thumbnail_path,
                "is_decorative": image.info.is_decorative,
                "extracted_at": image.extracted_at.isoformat(),
                "download_url": f"/api/images/download/{image.id}" if image.local_path else None,
                "thumbnail_url": f"/api/images/thumbnail/{image.id}" if image.thumbnail_path else None
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching images for content {content_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch images")

@router.get("/download/{image_id}")
async def download_image(
    image_id: int = FastAPIPath(..., description="ID of the image to download"),
    repo: ImageRepository = Depends(get_image_repository)
):
    """Download the full-size image file"""
    try:
        image = repo.get_image_by_id(image_id)
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")
        
        if not image.local_path or not Path(image.local_path).exists():
            raise HTTPException(status_code=404, detail="Image file not found")
        
        return FileResponse(
            path=image.local_path,
            filename=f"image_{image_id}.{image.info.image_format.value.lower()}",
            media_type=f"image/{image.info.image_format.value.lower()}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading image {image_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to download image")

@router.get("/thumbnail/{image_id}")
async def get_thumbnail(
    image_id: int = FastAPIPath(..., description="ID of the image to get thumbnail for"),
    repo: ImageRepository = Depends(get_image_repository)
):
    """Get the thumbnail version of an image"""
    try:
        image = repo.get_image_by_id(image_id)
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")
        
        if not image.thumbnail_path or not Path(image.thumbnail_path).exists():
            # Fall back to full image if thumbnail doesn't exist
            if image.local_path and Path(image.local_path).exists():
                return FileResponse(
                    path=image.local_path,
                    filename=f"thumb_{image_id}.{image.info.image_format.value.lower()}",
                    media_type=f"image/{image.info.image_format.value.lower()}"
                )
            else:
                raise HTTPException(status_code=404, detail="Image thumbnail not found")
        
        return FileResponse(
            path=image.thumbnail_path,
            filename=f"thumb_{image_id}.{image.info.image_format.value.lower()}",
            media_type=f"image/{image.info.image_format.value.lower()}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting thumbnail for image {image_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get thumbnail")

@router.get("/stats/{content_id}")
async def get_image_stats(
    content_id: int = FastAPIPath(..., description="ID of the scraped content"),
    repo: ImageRepository = Depends(get_image_repository)
):
    """Get image statistics for a specific scraped content"""
    try:
        images = repo.get_images_by_content_id(content_id)
        
        stats = {
            "total_images": len(images),
            "by_type": {},
            "by_format": {},
            "decorative_count": sum(1 for img in images if img.info.is_decorative),
            "content_count": sum(1 for img in images if not img.info.is_decorative),
            "total_size": sum(img.info.file_size for img in images if img.info.file_size),
            "downloaded_count": sum(1 for img in images if img.local_path)
        }
        
        # Count by type
        for image in images:
            type_key = image.info.image_type.value
            stats["by_type"][type_key] = stats["by_type"].get(type_key, 0) + 1
        
        # Count by format
        for image in images:
            format_key = image.info.image_format.value
            stats["by_format"][format_key] = stats["by_format"].get(format_key, 0) + 1
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting image stats for content {content_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get image statistics")

@router.get("/types")
async def get_image_types():
    """Get available image types"""
    return {
        "image_types": [{"value": t.value, "name": t.name} for t in ImageType],
        "image_formats": [{"value": f.value, "name": f.name} for f in ImageFormat]
    }
