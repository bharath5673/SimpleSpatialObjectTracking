import csv
import cv2
import numpy as np
import os
from PIL import Image
from ultralytics import YOLO

np.random.seed(42)
TRACK_COLORS = {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(5000)}
track_history = {}
map_track_history = {}
MAX_TRAIL = 20

def process_frame(image, model):
    results = model.track(image, verbose=False, persist=True, tracker="bytetrack.yaml")
    detections = []
    
    if not results or results[0].boxes is None or results[0].boxes.id is None:
        return detections

    for box, conf, cls, oid in zip(results[0].boxes.xyxy, 
                                   results[0].boxes.conf, 
                                   results[0].boxes.cls, 
                                   results[0].boxes.id):
        x1, y1, x2, y2 = map(int, box)
        detections.append({
            "id": int(oid),
            "bbox": (x1, y1, x2, y2),
            "class": results[0].names[int(cls)],
            "score": float(conf)
        })
    return detections

def draw_annotations(image, detections):
    global track_history
    img = image.copy()

    for det in detections:
        tid, (x1, y1, x2, y2), label, score = det["id"], det["bbox"], det["class"], det["score"]
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        txt = f"{label} {tid} ({score:.2f})"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), (0, 0, 0), -1)
        cv2.putText(img, txt, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        color = TRACK_COLORS.get(tid, (0, 255, 0))
        cv2.circle(img, (cx, cy), 4, color, -1)

        if tid not in track_history:
            track_history[tid] = []
        track_history[tid].append((cx, cy))
        if len(track_history[tid]) > MAX_TRAIL:
            track_history[tid].pop(0)

        pts = track_history[tid]
        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i], color, 2)

    return img


def load_object_locations(path):
    objects = []
    if os.path.exists(path):
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    obj_id = row.get("object_id") or row.get("id") or row.get("track_id")
                    lat = float(row.get("latitude", row.get("lat", "")))
                    lon = float(row.get("longitude", row.get("lon", "")))
                    frame = int(row.get("frame", 0)) if row.get("frame") else 0
                    objects.append({"id": obj_id, "lat": lat, "lon": lon, "frame": frame})
                except (ValueError, TypeError):
                    continue
    return objects


def plot_objects_on_map(map_image, objects, gps_center, map_width=640, map_height=720):
    img = Image.fromarray(cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    if not objects:
        result = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return result
    
    # Calculate bounds from actual object coordinates
    lats = [obj["lat"] for obj in objects]
    lons = [obj["lon"] for obj in objects]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    
    # Add 10% padding
    lat_padding = (lat_max - lat_min) * 0.1 if lat_max > lat_min else 0.0005
    lon_padding = (lon_max - lon_min) * 0.1 if lon_max > lon_min else 0.0005
    lat_min -= lat_padding
    lat_max += lat_padding
    lon_min -= lon_padding
    lon_max += lon_padding
    
    def project(lat, lon):
        px = int((lon - lon_min) / (lon_max - lon_min) * map_width)
        py = int((lat_max - lat) / (lat_max - lat_min) * map_height)
        return px, py
    
    for obj in objects:
        px, py = project(obj["lat"], obj["lon"])
        if 0 <= px < map_width and 0 <= py < map_height:
            draw.ellipse(
                (px - 6, py - 6, px + 6, py + 6),
                fill=(255, 0, 0),
                outline=(255, 255, 255),
                width=2
            )
            draw.text((px + 10, py - 10), str(obj["id"]), fill=(255, 255, 255))
    
    result = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return result


def plot_detected_on_map(map_image, detections, video_w, video_h, map_width, map_height, scale=0.8, rotation=0, offset_x=0, offset_y=0):
    """Plot detected objects from video onto map with trails, scaling, rotation, and pixel offsets.

    offset_x / offset_y are pixel translations applied after scaling+rotation to nudge points
    left/right/up/down to better align with the map image.
    """
    global map_track_history
    img = Image.fromarray(cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB))
    img_cv = np.array(img)

    # Map center
    center_x, center_y = map_width / 2.0, map_height / 2.0

    # Scale factors from video to map
    scale_x = map_width / float(video_w)
    scale_y = map_height / float(video_h)

    # Rotation in radians
    angle_rad = np.radians(rotation)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    def transform_point(map_x, map_y):
        """Apply scaling, rotation and final translation (offsets) to point"""
        # Translate to center
        x = map_x - center_x
        y = map_y - center_y

        # Apply uniform scale (preserve aspect by using average of scale_x/scale_y)
        x *= scale
        y *= scale

        # Apply rotation about center
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a

        # Translate back and apply user pixel offsets
        x_final = int(x_rot + center_x + offset_x)
        y_final = int(y_rot + center_y + offset_y)

        return x_final, y_final

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        tid = det["id"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Scale to map coordinates (map-space before transform)
        map_x = cx * scale_x
        map_y = cy * scale_y

        # Apply transformations
        map_x_t, map_y_t = transform_point(map_x, map_y)

        if 0 <= map_x_t < map_width and 0 <= map_y_t < map_height:
            # Track position
            if tid not in map_track_history:
                map_track_history[tid] = []
            map_track_history[tid].append((map_x_t, map_y_t))
            if len(map_track_history[tid]) > MAX_TRAIL:
                map_track_history[tid].pop(0)

            # Draw trail
            color = TRACK_COLORS.get(tid, (0, 255, 0))
            pts = map_track_history[tid]
            for i in range(1, len(pts)):
                cv2.line(img_cv, pts[i - 1], pts[i], color, 2)

            # Draw circle
            cv2.circle(img_cv, (int(map_x_t), int(map_y_t)), 5, color, -1)
            cv2.putText(img_cv, str(tid), (int(map_x_t) + 8, int(map_y_t) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    result = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    return result


if __name__ == "__main__":
    video_path = 'inputs/youtube_stream_20250321_151616.mp4'
    resize_w, resize_h = 1280, 720
    map_w, map_h = 960, 720
    gps_center = [53.345520, -6.264284]
    
    # Transformation parameters
    scale = 0.18  # Scale factor to shrink points towards center (0-1, smaller = more shrink)
    rotation = 100  # Rotation angle in degrees
    # Pixel offsets to nudge mapped points (positive x -> right, positive y -> down)
    offset_x = -5
    offset_y = 18
    
    model = YOLO("/home/bharath/Downloads/test_codes/models/yolov8n.pt")
    model.overrides.update({"conf": 0.3, "iou": 0.4, "classes": [0]})
    model.to("cuda")

    base_map = cv2.imread("outputs/saveGpsMap.png")
    if base_map is not None:
        base_map = cv2.resize(base_map, (map_w, map_h))
    else:
        base_map = np.ones((map_h, map_w, 3), dtype=np.uint8) * 200
    
    objects = load_object_locations("inputs/object_locations.csv")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    output_path = f"outputs/tracked_{os.path.basename(video_path)}"
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 25,
        (resize_w + map_w, resize_h)
    )

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (resize_w, resize_h))
        dets = process_frame(frame, model)
        frame_out = draw_annotations(frame, dets)
        
        # Plot detected objects on map with scaling, rotation and offsets
        map_with_all = plot_detected_on_map(base_map.copy(), dets, resize_w, resize_h, map_w, map_h, scale, rotation, offset_x, offset_y)
        
        combined = np.concatenate((frame_out, map_with_all), axis=1)
        out.write(combined)
        cv2.imshow("YOLO Track + GPS Map", combined)
        
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()