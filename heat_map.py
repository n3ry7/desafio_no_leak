import json
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

def parse_geojson(json_path):
    """Parse the GeoJSON file to extract person detection centroids."""
    with open(json_path, 'r') as file:
        data = json.load(file)
    hits = data.get('hits', {}).get('hits', [])
    person_detections = []
    for hit in hits:
        fields = hit.get('fields', {})
        deepstream_msgs = fields.get('deepstream-msg', [])
        for msg in deepstream_msgs:
            parts = msg.split('|')
            if len(parts) != 7:
                continue  # Skip invalid messages
            try:
                x_min = float(parts[1])
                y_min = float(parts[2])
                x_max = float(parts[3])
                y_max = float(parts[4])
                obj_type = parts[5]
            except (IndexError, ValueError):
                continue  # Skip invalid entries
            if obj_type.lower() == 'person':
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                person_detections.append((x_center, y_center))
    return np.array(person_detections)

def generate_heatmap(detections, width, height, sigma=15, cap_value=100):
    """Generate a heatmap from detection coordinates."""
    heatmap = np.zeros((height, width), dtype=np.float32)
    for x, y in detections:
        if 0 <= x < width and 0 <= y < height:
            heatmap[int(y), int(x)] += 1
    # Smooth the heatmap
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    # Cap the maximum value
    heatmap = np.clip(heatmap, None, cap_value)
    # Normalize to 0-255
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    return heatmap.astype(np.uint8)

def create_custom_colormap(alpha=0.45):
    """Create a custom RGBA colormap with full transparency at zero intensity."""
    colormap = np.zeros((256, 4), dtype=np.uint8)
    
    # Special case: full transparency for zero values
    colormap[0] = [0, 0, 0, 0]  # BGRA format
    
    # Define color transitions in BGRA order (with alpha)
    blue = np.array([255, 0, 0, int(255 * alpha)])
    green = np.array([0, 255, 0, int(255 * alpha)])
    yellow = np.array([0, 255, 255, int(255 * alpha)])
    orange = np.array([0, 165, 255, int(255 * alpha)])
    red = np.array([0, 0, 255, int(255 * alpha)])

    # Thresholds (percentage of 255)
    thrs = [int(255 * p) for p in [0.10, 0.25, 0.50, 0.70]]
    thrs1, thrs2, thrs3, thrs4 = thrs

    for i in range(1, 256):  # Start from 1 since 0 is handled
        if i <= thrs1:
            ratio = i / thrs1
            colormap[i] = (1 - ratio) * blue + ratio * green
        elif i <= thrs2:
            ratio = (i - thrs1) / (thrs2 - thrs1)
            colormap[i] = (1 - ratio) * green + ratio * yellow
        elif i <= thrs3:
            ratio = (i - thrs2) / (thrs3 - thrs2)
            colormap[i] = (1 - ratio) * yellow + ratio * orange
        elif i <= thrs4:
            ratio = (i - thrs3) / (thrs4 - thrs3)
            colormap[i] = (1 - ratio) * orange + ratio * red
        else:
            colormap[i] = red

    return colormap

def apply_custom_colormap(heatmap, colormap):
    """Apply the custom colormap to the heatmap."""
    return colormap[heatmap]

def overlay_heatmap(rgb_image, heatmap_rgba):
    """Overlay the heatmap onto the RGB image with alpha blending."""
    # Convert image to BGRA
    image_bgra = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2BGRA)
    # Extract alpha from heatmap
    heatmap_alpha = heatmap_rgba[:, :, 3] / 255.0
    heatmap_alpha = np.expand_dims(heatmap_alpha, axis=-1)
    # Create masked heatmap
    masked_heatmap = heatmap_rgba * heatmap_alpha
    # Create masked image
    masked_image = image_bgra * (1 - heatmap_alpha)
    # Blend the images
    blended = (masked_heatmap + masked_image).astype(np.uint8)
    return cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)

def generate_overlayed_image(geojson_path, image_path, target_size=(708, 480)):
    """High-level function to generate the overlayed heatmap image."""
    detections = parse_geojson(geojson_path)
    if len(detections) == 0:
        print("No person detections found.")
        return None
    # Load and resize image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    image = cv2.resize(image, target_size)
    # Generate heatmap
    heatmap = generate_heatmap(detections, target_size[0], target_size[1])
    # Create and apply colormap
    colormap = create_custom_colormap()
    heatmap_rgba = apply_custom_colormap(heatmap, colormap)
    # Resize heatmap to match image dimensions
    heatmap_rgba = cv2.resize(heatmap_rgba, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    # Overlay heatmap
    overlayed_image = overlay_heatmap(image, heatmap_rgba)
    return overlayed_image

def main():
    """Test the pipeline with default paths."""
    overlayed = generate_overlayed_image('response.json', 'image.png')
    if overlayed is not None:
        cv2.imwrite('masked_heatmap_output.png', overlayed)
        print("Overlayed image saved successfully.")

if __name__ == "__main__":
    main()
