from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi import status
import numpy as np
import cv2
import tempfile
import os
import heat_map

app = FastAPI()

# File size limits (15MB for JSON, 5MB for images)
MAX_JSON_SIZE = 15_000_000  # 15MB
MAX_IMAGE_SIZE = 5_000_000  # 5MB

def process_uploaded_files(image_file: UploadFile, json_file: UploadFile):
    """Handle file processing with proper cleanup"""
    # Check file sizes first before processing
    json_content = json_file.file.read()
    if len(json_content) > MAX_JSON_SIZE:
        raise ValueError(f"JSON file too large (max {MAX_JSON_SIZE} bytes allowed)")

    image_data = image_file.file.read()
    if len(image_data) > MAX_IMAGE_SIZE:
        raise ValueError(f"Image file too large (max {MAX_IMAGE_SIZE} bytes allowed)")

    # Process image file
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file")
    img = cv2.resize(img, (708, 480))

    # Process JSON file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_json:
        temp_json.write(json_content)
        temp_json_path = temp_json.name
    
    try:
        detections = heat_map.parse_geojson(temp_json_path)
    finally:
        os.unlink(temp_json_path)

    return img, detections

@app.post("/generate-overlay")
async def generate_heatmap_overlay(
    image: UploadFile = File(..., description="Upload image file (JPG/PNG)"),
    json_data: UploadFile = File(..., description="Upload JSON detection data")
):
    """
    Endpoint that accepts an image and JSON data, returns heatmap-overlayed image
    """
    try:
        # Validate file types
        if not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image file type"
            )
        if not json_data.filename.endswith('.json'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="JSON file required"
            )

        # Process uploaded files
        try:
            img, detections = process_uploaded_files(image, json_data)
        except ValueError as e:
            if "too large" in str(e).lower():
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=str(e)
                )
            raise

        if len(detections) == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No person detections found in JSON data"
            )

        # Generate heatmap overlay
        heatmap = heat_map.generate_heatmap(detections, 708, 480)
        colormap = heat_map.create_custom_colormap()
        heatmap_rgba = heat_map.apply_custom_colormap(heatmap, colormap)
        result_image = heat_map.overlay_heatmap(img, heatmap_rgba)

        # Encode response
        _, encoded_img = cv2.imencode(".png", result_image)
        return Response(
            content=encoded_img.tobytes(),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=overlay.png"}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
