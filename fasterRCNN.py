import cv2
import numpy as np
import torch
import cvzone
from torchvision import transforms
import torchvision
from sort import Sort
import time

# ——— Performance Optimization Settings ———
PROCESSING_WIDTH = 640  # Smaller frame size for processing
PROCESSING_HEIGHT = 360
DISPLAY_WIDTH = 1020    # Original display size
DISPLAY_HEIGHT = 500
DETECTION_INTERVAL = 3  # Only run detection every N frames
CONFIDENCE_THRESHOLD = 0.5

# ——— Model & Tracker Setup ———
# Load model only when needed and use eval mode for inference
def get_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    # If CUDA is available, move model to GPU
    if torch.cuda.is_available():
        model = model.cuda()
    return model

model = None  # Will be loaded in main()
tracker = Sort(max_age=20, min_hits=3)  # Increased max_age for smoother tracking
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300,
                                                  varThreshold=16,
                                                  detectShadows=False)  # Disable shadows for speed

# ——— Globals for Region Drawing ———
drawing = False
start_point = end_point = None
drawing_region = None
area1 = []  # IN region
area2 = []  # OUT region

# Capacity alert settings
headcount_limit = 5  # Default capacity limit
alert_active = False
last_alert_time = 0
alert_cooldown = 3  # Seconds between alerts

# Preprocessing transformation - do once
preprocess = transforms.Compose([
    transforms.ToTensor(),
])

# ——— Load Classes ———
def load_class_list(path):
    try:
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Warning: {path} not found. Using default COCO classes.")
        return ["background", "person"]  # Fallback to basic classes

classnames = load_class_list('classes.txt')

# ——— Mouse Callback ———
def draw_rectangle(event, x, y, flags, param):
    global drawing, start_point, end_point, drawing_region, area1, area2
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point, end_point = (x,y), (x,y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end_point = (x,y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x,y)
        region = [
            start_point,
            (end_point[0], start_point[1]),
            end_point,
            (start_point[0], end_point[1])
        ]
        if drawing_region == 'in':
            area1[:] = region.copy()
            print("Set IN region:", area1)
        elif drawing_region == 'out':
            area2[:] = region.copy()
            print("Set OUT region:", area2)
        start_point = end_point = None
        drawing_region = None

# ——— Pre‑filter Frame ———
def apply_filters(frame):
    # Reduced filtering - just enough to remove noise
    return cv2.GaussianBlur(frame, (3,3), 1.2)

# ——— Object Detection ———
@torch.no_grad()  # Disable gradient calculation for inference
def detect_people(frame, model):
    # Convert to tensor
    img_tensor = preprocess(frame)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
    
    # Run inference
    outputs = model([img_tensor])[0]
    
    # Get boxes, scores, and labels
    boxes = outputs['boxes'].cpu().numpy().astype(int)
    scores = outputs['scores'].cpu().numpy()
    labels = outputs['labels'].cpu().numpy()
    
    # Filter by person class (1) and confidence
    people_dets = []
    for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
        if label == 1 and score >= CONFIDENCE_THRESHOLD:
            people_dets.append([x1, y1, x2, y2])
    
    return np.array(people_dets) if people_dets else np.empty((0, 4))

# ——— Frame Processor ———
def process_frame(frame, motion_mask, tracker, area1, area2, states, cnt_in, cnt_out, detect=False, dets=None):
    h, w = frame.shape[:2]
    processed_frame = frame.copy()
    
    # Create region masks (only once if regions haven't changed)
    in_mask = np.zeros((h, w), dtype=np.uint8)
    out_mask = np.zeros((h, w), dtype=np.uint8)
    
    if len(area1) == 4:
        cv2.fillPoly(in_mask, [np.array(area1, np.int32)], 255)
    if len(area2) == 4:
        cv2.fillPoly(out_mask, [np.array(area2, np.int32)], 255)
    
    # Update tracker with detections
    if dets is not None and dets.size > 0:
        tracked = tracker.update(dets)
    else:
        tracked = tracker.update(np.empty((0, 4)))
    
    # Draw & count
    for obj in tracked:
        arr = obj.astype(int)
        x1, y1, x2, y2 = arr[:4]
        obj_id = int(arr[-1])
        
        # Draw box & ID
        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cvzone.putTextRect(processed_frame, f'ID {obj_id}', (x1, y1-10), 1, 1)
        
        # Count region transitions
        roi = motion_mask[y1:y2, x1:x2]
        if roi.size > 0:  # Check if ROI is valid
            in_ov = cv2.countNonZero(cv2.bitwise_and(roi, in_mask[y1:y2, x1:x2]))
            out_ov = cv2.countNonZero(cv2.bitwise_and(roi, out_mask[y1:y2, x1:x2]))
            
            prev = states.get(obj_id)
            if prev is None:
                if in_ov > 100:  # Lowered threshold for better detection
                    states[obj_id] = 'in'
                elif out_ov > 100:
                    states[obj_id] = 'out'
            else:
                if prev == 'in' and out_ov > 100:
                    cnt_out.add(obj_id)
                    del states[obj_id]
                elif prev == 'out' and in_ov > 100:
                    cnt_in.add(obj_id)
                    del states[obj_id]
    
    return processed_frame, len(cnt_in), len(cnt_out)

# ——— Main Loop ———
def main():
    global drawing_region, start_point, end_point, headcount_limit, alert_active, last_alert_time, model
    
    # Load model only when needed
    print("Loading detection model...")
    model = get_model()
    print("Model loaded!")
    
    cv2.namedWindow('people_counter')
    cv2.setMouseCallback('people_counter', draw_rectangle)
    
    states = {}
    cnt_in = set()
    cnt_out = set()
    print("Press 'i' (IN), 'o' (OUT), 'l' (set Limit), ESC to exit.")
    
    cap = cv2.VideoCapture(0)
    
    frame_count = 0
    last_detections = np.empty((0, 4))
    fps_time = time.time()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame for display
        display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        
        # Resize frame for processing (smaller = faster)
        process_frame = cv2.resize(frame, (PROCESSING_WIDTH, PROCESSING_HEIGHT))
        process_frame = apply_filters(process_frame)
        
        # Update frame counter
        frame_count += 1
        
        # Calculate FPS
        if time.time() - fps_time >= 1.0:
            fps = frame_count
            frame_count = 0
            fps_time = time.time()
        
        # Apply background subtraction
        motion = bg_subtractor.apply(process_frame)
        _, motion = cv2.threshold(motion, 240, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Only run detection every DETECTION_INTERVAL frames
        run_detection = (frame_count % DETECTION_INTERVAL == 0)
        
        if run_detection:
            # Run people detection
            detections = detect_people(process_frame, model)
            last_detections = detections
        
        # Process frame with detections
        result_frame, in_count, out_count = process_frame(
            display_frame,
            cv2.resize(motion, (DISPLAY_WIDTH, DISPLAY_HEIGHT)),
            tracker,
            area1, area2,
            states, cnt_in, cnt_out,
            detect=run_detection,
            dets=last_detections
        )
        
        # Draw regions
        if start_point and end_point:
            cv2.rectangle(result_frame, start_point, end_point, (0, 255, 255), 2)
        if len(area1) == 4:
            cv2.polylines(result_frame, [np.array(area1)], True, (0, 255, 0), 2)
        if len(area2) == 4:
            cv2.polylines(result_frame, [np.array(area2)], True, (0, 0, 255), 2)
        
        # Calculate current headcount
        current_headcount = max(0, in_count - out_count)  # Prevent negative values
        
        # Display information
        cv2.putText(result_frame, f'In: {in_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(result_frame, f'Out: {out_count}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(result_frame, f'Headcount: {current_headcount}', (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        cv2.putText(result_frame, f'Limit: {headcount_limit}', (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
        cv2.putText(result_frame, f'FPS: {fps}', (DISPLAY_WIDTH - 120, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        # Check if headcount exceeds limit
        current_time = time.time()
        if current_headcount > headcount_limit:
            if not alert_active or (current_time - last_alert_time > alert_cooldown):
                print(f"ALERT: Headcount ({current_headcount}) exceeds limit ({headcount_limit})!")
                alert_active = True
                last_alert_time = current_time
        else:
            alert_active = False
        
        # Display alert if active
        if alert_active:
            # Create red alert overlay
            alert_overlay = result_frame.copy()
            cv2.rectangle(alert_overlay, (0, 0), (result_frame.shape[1], result_frame.shape[0]), (0, 0, 255), -1)
            result_frame = cv2.addWeighted(result_frame, 0.7, alert_overlay, 0.3, 0)
            
            # Add warning text
            cv2.putText(result_frame, "MAXIMUM CAPACITY EXCEEDED!", 
                      (int(result_frame.shape[1]/2) - 300, int(result_frame.shape[0]/2)), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        
        cv2.imshow('people_counter', result_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('i'):
            drawing_region = 'in'
        elif key == ord('o'):
            drawing_region = 'out'
        elif key == ord('l'):
            try:
                new_limit = int(input("Enter new headcount limit: "))
                if new_limit >= 0:
                    headcount_limit = new_limit
                    print(f"Headcount limit set to {headcount_limit}")
                else:
                    print("Limit must be a positive number.")
            except ValueError:
                print("Please enter a valid number.")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()