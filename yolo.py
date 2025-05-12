import cv2
import numpy as np
import time
from ultralytics import YOLO
from sort import Sort
import cvzone

# Initialize models
model         = YOLO('yolov8s-seg.pt')
tracker       = Sort()
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500,
                                                   varThreshold=16,
                                                   detectShadows=True)

# Globals for region drawing
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

def draw_rectangle(event, x, y, flags, param):
    global drawing, start_point, end_point, drawing_region, area1, area2
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True; start_point = (x,y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end_point = (x,y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False; end_point = (x,y)
        region = [
            start_point,
            (end_point[0], start_point[1]),
            end_point,
            (start_point[0], end_point[1])
        ]
        if drawing_region == 'in':
            area1[:] = region.copy(); print("Set IN region:", area1)
        elif drawing_region == 'out':
            area2[:] = region.copy(); print("Set OUT region:", area2)

def load_class_list(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]

def apply_filters(frame, k=3):
    frame = cv2.medianBlur(frame, k)
    frame = cv2.GaussianBlur(frame, (k,k), 1.2)
    return frame

def process_frame(frame, classes, tracker, area1, area2, states, cnt_in, cnt_out):
    # 1) Motion mask
    motion = bg_subtractor.apply(frame)
    _, motion = cv2.threshold(motion, 254, 255, cv2.THRESH_BINARY)
    motion = cv2.morphologyEx(
        motion, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)),
        iterations=2
    )

    # 2) YOLOv8 detection + seg
    res = model.predict(frame, verbose=False)[0]
    boxes = res.boxes.data.cpu().numpy().astype(int)
    masks = res.masks.data.cpu().numpy() if res.masks else []

    # 3) Filter by motion overlap
    dets = []
    for (x1,y1,x2,y2,_,cls_id), seg in zip(boxes, masks):
        if classes[cls_id] != 'person': continue
        roi = motion[y1:y2, x1:x2]
        if cv2.countNonZero(roi) < 0.2 * roi.size: continue
        dets.append([x1,y1,x2,y2])
    dets = np.array(dets)

    # 4) SORT tracking (guard empty)
    if dets.shape[0] == 0:
        tracked = np.empty((0,5))
    else:
        tracked = tracker.update(dets)

    # 5) Region masks
    h, w = frame.shape[:2]
    in_mask  = np.zeros((h,w), dtype=np.uint8)
    out_mask = np.zeros((h,w), dtype=np.uint8)
    if len(area1)==4:
        cv2.fillPoly(in_mask, [np.array(area1, np.int32)], 255)
    if len(area2)==4:
        cv2.fillPoly(out_mask,[np.array(area2, np.int32)], 255)

    overlay = frame.copy()
    alpha = 0.5

    for obj in tracked:
        arr = obj.astype(int)
        x1, y1, x2, y2 = arr[:4]
        obj_id = int(arr[-1])  # unpack last column as ID

        # compute overlaps
        roi = motion[y1:y2, x1:x2]
        in_ov  = cv2.countNonZero(cv2.bitwise_and(roi, in_mask[y1:y2, x1:x2]))
        out_ov = cv2.countNonZero(cv2.bitwise_and(roi, out_mask[y1:y2, x1:x2]))

        prev = states.get(obj_id)
        if prev is None:
            if in_ov > 300:  states[obj_id] = 'in'
            elif out_ov > 300: states[obj_id] = 'out'
        else:
            if prev=='in' and out_ov>300:
                cnt_out.add(obj_id); del states[obj_id]
            elif prev=='out' and in_ov>300:
                cnt_in.add(obj_id);  del states[obj_id]

        # draw
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
        cvzone.putTextRect(frame, f'ID {obj_id}', (x1,y1-10), 1, 1)

    frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
    return frame, len(cnt_in), len(cnt_out)

def main():
    global drawing_region, start_point, end_point, headcount_limit, alert_active, last_alert_time
    classes     = load_class_list('coco.txt')
    states      = {}
    counter_in  = set()
    counter_out = set()

    cv2.namedWindow('people_counter')
    cv2.setMouseCallback('people_counter', draw_rectangle)
    print("Press 'i' (IN), 'o' (OUT), 'l' (set Limit), ESC to exit.")

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (1020,500))
        frame = apply_filters(frame)

        # show drawing overlay
        if start_point and end_point:
            cv2.rectangle(frame, start_point, end_point, (0,255,255), 2)
        if len(area1)==4:
            cv2.polylines(frame, [np.array(area1)], True, (0,255,0), 2)
        if len(area2)==4:
            cv2.polylines(frame, [np.array(area2)], True, (0,0,255), 2)

        out_frame, in_c, out_c = process_frame(
            frame, classes, tracker,
            area1, area2, states,
            counter_in, counter_out
        )

        # Calculate current headcount
        current_headcount = in_c - out_c
        
        # Check if headcount exceeds limit
        current_time = time.time()
        if current_headcount > headcount_limit:
            if not alert_active or (current_time - last_alert_time > alert_cooldown):
                print(f"ALERT: Headcount ({current_headcount}) exceeds limit ({headcount_limit})!")
                alert_active = True
                last_alert_time = current_time
        else:
            alert_active = False

        # Regular info display
        cv2.putText(out_frame, f'In: {in_c}', (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(out_frame, f'Out: {out_c}', (20,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(out_frame, f'Headcount in event area: {current_headcount}', (20,130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100,100,100), 2)
        cv2.putText(out_frame, f'Limit: {headcount_limit}', (20,170),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,100,255), 2)
                    
        # Display alert if active
        if alert_active:
            # Create red alert overlay
            alert_overlay = out_frame.copy()
            cv2.rectangle(alert_overlay, (0, 0), (out_frame.shape[1], out_frame.shape[0]), (0, 0, 255), -1)
            out_frame = cv2.addWeighted(out_frame, 0.7, alert_overlay, 0.3, 0)
            
            # Add warning text
            cv2.putText(out_frame, "MAXIMUM CAPACITY EXCEEDED!", 
                      (int(out_frame.shape[1]/2) - 300, int(out_frame.shape[0]/2)), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        cv2.imshow('people_counter', out_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if   key==27: break
        elif key==ord('i'):
            drawing_region='in';  print("Draw IN region.")
        elif key==ord('o'):
            drawing_region='out'; print("Draw OUT region.")
        elif key==ord('l'):
            try:
                new_limit = int(input("Enter new headcount limit: "))
                if new_limit >= 0:
                    headcount_limit = new_limit
                    print(f"Headcount limit set to {headcount_limit}")
                     # Check if headcount exceeds limit
                    current_time = time.time()
                    if current_headcount > headcount_limit:
                        if not alert_active or (current_time - last_alert_time > alert_cooldown):
                            print(f"ALERT: Headcount ({current_headcount}) exceeds limit ({headcount_limit})!")
                            alert_active = True
                            last_alert_time = current_time
                        else:
                            alert_active = False
                else:
                    print("Limit must be a positive number.")
            except ValueError:
                print("Please enter a valid number.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()