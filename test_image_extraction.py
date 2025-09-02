#!/usr/bin/env python3
import sys
sys.path.insert(0, '/app/backend')
from bs4 import BeautifulSoup
from src.infrastructure.image_processor import ImageProcessor

# Create a simple HTML with an image
html = '<html><body><img src="test.jpg" alt="test image"></body></html>'
soup = BeautifulSoup(html, 'html.parser')

processor = ImageProcessor()
print('✅ ImageProcessor created')

images = processor.extract_images_from_html(soup, 'https://www.example.com', 'example')
print(f'📊 Extracted {len(images)} images')
for img in images:
    print(f'  - {img.url} ({img.alt_text})')

if __name__ == "__main__":
    print("Image extraction test completed")
        
        if result.status.name == "SUCCESS" and result.content:
            print(f"✅ Scraping successful!")
            print(f"📄 Title: {result.content.title}")
            print(f"📷 Images found: {len(result.content.images) if result.content.images else 0}")
            
            # Save to database
            content_id = repository.save_scraped_content(result.content)
            
            if content_id:
                print(f"💾 Saved to database with ID: {content_id}")
                
                # Fetch images from database to verify
                saved_images = repository.get_images_for_content(content_id)
                print(f"🔍 Verified {len(saved_images)} images in database")
                
                # Print image details
                for i, image in enumerate(saved_images[:3], 1):  # Show first 3
                    print(f"  {i}. {image.image_type.value} - {image.url}")
                    if image.alt_text:
                        print(f"     Alt: {image.alt_text[:50]}...")
            
        else:
            print(f"❌ Scraping failed: {result.error_message}")
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_image_extraction())
