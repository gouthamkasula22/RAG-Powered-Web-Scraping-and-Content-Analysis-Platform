#!/usr/bin/env python3
"""
Direct test of image extraction functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.src.infrastructure.image_processor import ImageProcessor, HAS_IMAGE_DEPS
import requests
from bs4 import BeautifulSoup

def test_image_extraction():
    print(f'Dependencies available: {HAS_IMAGE_DEPS}')
    print('Testing image extraction on BBC directly...')
    
    try:
        response = requests.get('https://www.bbc.com')
        print(f'Status: {response.status_code}')
        
        soup = BeautifulSoup(response.text, 'html.parser')
        print(f'HTML parsed, found {len(soup.find_all("img"))} img tags')
        
        processor = ImageProcessor()
        images = processor.extract_images_from_html(soup, 'https://www.bbc.com', 'test_bbc')
        print(f'Images processed: {len(images)}')
        
        for i, img in enumerate(images[:5]):
            alt_text = img.info.alt_text[:50] if img.info.alt_text else 'None'
            print(f'{i+1}. {img.info.original_url}')
            print(f'    Type: {img.info.image_type}')
            print(f'    Alt: {alt_text}')
            print(f'    Size: {img.info.width}x{img.info.height}' if img.info.width else '    Size: Unknown')
            print()
            
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_image_extraction()
