import pytest
from fastapi.testclient import TestClient
from fastapi import status
from api_main import app
import os
import cv2
import numpy as np

client = TestClient(app)

@pytest.fixture
def test_image():
    # Create a simple test image (white 708x480 image)
    img = np.ones((480, 708, 3), dtype=np.uint8) * 255
    _, img_encoded = cv2.imencode(".png", img)
    return img_encoded.tobytes()

@pytest.fixture
def test_json():
    # Create minimal valid JSON data with one person detection
    return b'''{
        "hits": {
            "hits": [
                {
                    "fields": {
                        "deepstream-msg": ["1|100|100|200|200|person|region"]
                    }
                }
            ]
        }
    }'''

def test_successful_overlay(test_image, test_json):
    files = {
        "image": ("test.png", test_image, "image/png"),
        "json_data": ("test.json", test_json, "application/json")
    }
    
    response = client.post("/generate-overlay", files=files)
    
    assert response.status_code == status.HTTP_200_OK
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 0
    
    # Verify the returned image is valid
    img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (480, 708, 3)

def test_invalid_image_format(test_json):
    files = {
        "image": ("test.txt", b"not an image", "text/plain"),
        "json_data": ("test.json", test_json, "application/json")
    }
    
    response = client.post("/generate-overlay", files=files)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Invalid image file type" in response.text

def test_invalid_json_format(test_image):
    files = {
        "image": ("test.png", test_image, "image/png"),
        "json_data": ("test.txt", b"not json", "text/plain")
    }
    
    response = client.post("/generate-overlay", files=files)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "JSON file required" in response.text

def test_no_detections(test_image):
    # JSON with no person detections
    empty_json = b'''{
        "hits": {
            "hits": [
                {
                    "fields": {
                        "deepstream-msg": ["1|100|100|200|200|car|region"]
                    }
                }
            ]
        }
    }'''
    
    files = {
        "image": ("test.png", test_image, "image/png"),
        "json_data": ("test.json", empty_json, "application/json")
    }
    
    response = client.post("/generate-overlay", files=files)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "No person detections" in response.text

def test_missing_fields():
    # Test with missing image field
    response = client.post("/generate-overlay", files={"json_data": ("test.json", b"{}", "application/json")})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    # Test with missing json field
    response = client.post("/generate-overlay", files={"image": ("test.png", b"", "image/png")})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_large_file_handling(test_json):
    # Create a 10MB test file (larger than our 5MB image limit)
    large_img_data = os.urandom(10_000_000)  # 10MB
    
    files = {
        "image": ("large.png", large_img_data, "image/png"),
        "json_data": ("test.json", test_json, "application/json")
    }
    
    response = client.post("/generate-overlay", files=files)
    assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    assert "too large" in response.text.lower()

def test_large_json_handling(test_image):
    # Create a 20MB JSON file (larger than our 15MB limit)
    large_json_data = os.urandom(20_000_000)  # 20MB
    
    files = {
        "image": ("test.png", test_image, "image/png"),
        "json_data": ("large.json", large_json_data, "application/json")
    }
    
    response = client.post("/generate-overlay", files=files)
    assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    assert "too large" in response.text.lower()
