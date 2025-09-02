"""
Database migration to add images table for storing extracted image metadata
"""
import sqlite3
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

MIGRATION_VERSION = "001_add_images_table"

def upgrade(db_path: str) -> bool:
    """
    Add images table to store extracted image metadata
    
    Args:
        db_path: Path to the database file
        
    Returns:
        bool: True if migration successful, False otherwise
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create images table
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
        
        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_images_scraped_content_id 
            ON images(scraped_content_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_images_type 
            ON images(image_type)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_images_url 
            ON images(url)
        """)
        
        # Add migration record
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS migrations (
                version TEXT PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            INSERT OR IGNORE INTO migrations (version) VALUES (?)
        """, (MIGRATION_VERSION,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Migration {MIGRATION_VERSION} applied successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration {MIGRATION_VERSION} failed: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False

def downgrade(db_path: str) -> bool:
    """
    Remove images table and related structures
    
    Args:
        db_path: Path to the database file
        
    Returns:
        bool: True if downgrade successful, False otherwise
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Drop images table and indexes
        cursor.execute("DROP INDEX IF EXISTS idx_images_scraped_content_id")
        cursor.execute("DROP INDEX IF EXISTS idx_images_type")
        cursor.execute("DROP INDEX IF EXISTS idx_images_url")
        cursor.execute("DROP TABLE IF EXISTS images")
        
        # Remove migration record
        cursor.execute("DELETE FROM migrations WHERE version = ?", (MIGRATION_VERSION,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Migration {MIGRATION_VERSION} downgraded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration {MIGRATION_VERSION} downgrade failed: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False

def check_migration_status(db_path: str) -> bool:
    """
    Check if this migration has been applied
    
    Args:
        db_path: Path to the database file
        
    Returns:
        bool: True if migration has been applied, False otherwise
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if migrations table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='migrations'
        """)
        
        if not cursor.fetchone():
            conn.close()
            return False
        
        # Check if this specific migration exists
        cursor.execute("""
            SELECT version FROM migrations WHERE version = ?
        """, (MIGRATION_VERSION,))
        
        result = cursor.fetchone() is not None
        conn.close()
        return result
        
    except Exception as e:
        logger.error(f"❌ Error checking migration status: {str(e)}")
        if 'conn' in locals():
            conn.close()
        return False

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) < 2:
        print("Usage: python add_images_table.py <database_path>")
        sys.exit(1)
    
    db_path = sys.argv[1]
    
    if check_migration_status(db_path):
        print(f"Migration {MIGRATION_VERSION} already applied")
    else:
        success = upgrade(db_path)
        if success:
            print(f"Migration {MIGRATION_VERSION} applied successfully")
        else:
            print(f"Migration {MIGRATION_VERSION} failed")
            sys.exit(1)
