import cv2
import numpy as np
import torch
import cvzone
from torchvision import transforms
import torchvision
from sort import Sort
import time
# ——— Model & Tracker Setup ———
model         = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
tracker       = Sort()
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500,
                                                   varThreshold=16,
                                                   detectShadows=True)

# ——— Globals for Region Drawing ———
# drawing = False
# start_point = None
# end_point = None
# drawing_region = None
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
area1 = []  # IN region
area2 = []  # OUT region

# ——— Load Classes ———
def load_class_list(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]
classnames = load_class_list('classes.txt')

# ——— Mouse Callback ———
def draw_rectangle(event, x, y, flags, param):
    global drawing, start_point, end_point, drawing_region, area1, area2
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True; start_point, end_point = (x,y), (x,y)
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
            area1[:] = region.copy()
            print("Set IN region:", area1)
        elif drawing_region == 'out':
            area2[:] = region.copy()
            print("Set OUT region:", area2)
        start_point = end_point = None
        drawing_region = None

# ——— Pre‑filter Frame ———
def apply_filters(frame, k=3):
    frame = cv2.medianBlur(frame, k)
    return cv2.GaussianBlur(frame, (k,k), 1.2)

# ——— Frame Processor ———
def process_frame(frame, tracker, area1, area2, states, cnt_in, cnt_out):
    # 1) Motion mask
    motion = bg_subtractor.apply(frame)
    _, motion = cv2.threshold(motion, 254, 255, cv2.THRESH_BINARY)
    motion = cv2.morphologyEx(
        motion, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)),
        iterations=2
    )

    # 2) Faster R‑CNN inference
    img_tensor = transforms.ToTensor()(frame)
    with torch.no_grad():
        outputs = model([img_tensor])[0]

    boxes  = outputs['boxes'].cpu().numpy().astype(int)
    scores = outputs['scores'].cpu().numpy()
    labels = outputs['labels'].cpu().numpy()

    # 3) Filter by person label + score + motion overlap
    dets = []
    for (x1,y1,x2,y2), s, lbl in zip(boxes, scores, labels):
        if s < 0.3 or lbl != 1:
            continue
        roi = motion[y1:y2, x1:x2]
        if roi.size == 0 or cv2.countNonZero(roi) < 0.05*roi.size:
            continue
        dets.append([x1,y1,x2,y2])
    dets = np.array(dets)

    # 4) SORT tracking
    tracked = tracker.update(dets) if dets.size else np.empty((0,5))

    # 5) Build region masks
    h, w = frame.shape[:2]
    in_mask  = np.zeros((h,w), dtype=np.uint8)
    out_mask = np.zeros((h,w), dtype=np.uint8)
    if len(area1)==4: cv2.fillPoly(in_mask, [np.array(area1, np.int32)], 255)
    if len(area2)==4: cv2.fillPoly(out_mask,[np.array(area2, np.int32)], 255)

    overlay = frame.copy()
    alpha = 0.5

    # 6) Draw & count
    for obj in tracked:
        arr         = obj.astype(int)
        x1, y1, x2, y2 = arr[:4]
        obj_id      = int(arr[-1])

        # draw box & ID
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
        cvzone.putTextRect(frame, f'ID {obj_id}', (x1, y1-10), 1, 1)

        # count region transitions
        roi   = motion[y1:y2, x1:x2]
        in_ov = cv2.countNonZero(cv2.bitwise_and(roi, in_mask[y1:y2, x1:x2]))
        out_ov= cv2.countNonZero(cv2.bitwise_and(roi, out_mask[y1:y2, x1:x2]))
        prev  = states.get(obj_id)
        if prev is None:
            if in_ov  > 300: states[obj_id] = 'in'
            elif out_ov > 300: states[obj_id] = 'out'
        else:
            if prev=='in' and out_ov>300:
                cnt_out.add(obj_id); del states[obj_id]
            elif prev=='out' and in_ov>300:
                cnt_in.add(obj_id);  del states[obj_id]

    frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
    return frame, len(cnt_in), len(cnt_out)

# ——— Main Loop ———
def main():
    global drawing_region, start_point, end_point, headcount_limit, alert_active, last_alert_time

    cv2.namedWindow('people_counter')
    cv2.setMouseCallback('people_counter', draw_rectangle)

    states    = {}
    cnt_in    = set()
    cnt_out   = set()
    print("Press 'i' (IN), 'o' (OUT), 'l' (set Limit), ESC to exit.")


     # Calculate current headcount
    current_headcount = in_c - out_c

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (1020,500))
        frame = apply_filters(frame)

        # region‑drawing overlay
        if start_point and end_point:
            cv2.rectangle(frame, start_point, end_point, (0,255,255), 2)
        if len(area1)==4:
            cv2.polylines(frame, [np.array(area1)], True, (0,255,0), 2)
        if len(area2)==4:
            cv2.polylines(frame, [np.array(area2)], True, (0,0,255), 2)

        out_frame, in_c, out_c = process_frame(
            frame, tracker,
            area1, area2,
            states, cnt_in, cnt_out
        )

        cv2.putText(out_frame, f'In: {in_c}',  (20, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
        cv2.putText(out_frame, f'Out: {out_c}', (20, 90),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
        cv2.putText(out_frame, f'Headcount in event area: {in_c - out_c}', (20,130),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(100,100,100),2)

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
        if key==27: break
        elif key==ord('i'):
            drawing_region='in'
        elif key==ord('o'):
            drawing_region='out'
        elif key==ord('l'):
            try:
                new_limit = int(input("Enter new headcount limit: "))
                if new_limit >= 0:
                    headcount_limit = new_limit
                    print(f"Headcount limit set to {headcount_limit}")
                        # Check if headcount exceeds limit
                    current_time = time.time()
                    if (in_c - out_c) > headcount_limit:
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