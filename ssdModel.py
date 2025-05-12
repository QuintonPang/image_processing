
import cv2
import numpy as np
import torch
import torchvision
import cvzone
from sort import Sort            # local SORT implementation
import time

# ─────────────────────────── 1. SSD‑300 DETECTOR ──────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ssd_model = torchvision.models.detection.ssd300_vgg16(weights="DEFAULT")
ssd_model.eval().to(device)

from torchvision import transforms as T
tfm = T.Compose([T.ToTensor()])           # BGR uint8 → RGB float tensor [0,1]

# ─────────────────────────── 2. TRACKER & BGS ──────────────────────────────
tracker       = Sort()
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=16, detectShadows=True
)

# ─────────────────────────── 3. DRAWING GLOBALS ────────────────────────────
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

start_pt = end_pt = None
draw_mode = None                         # 'in' | 'out' | None
area_in, area_out = [], []               # 4‑point rectangles

# ─────────────────────────── 4. MOUSE CALLBACK ─────────────────────────────
def draw_rect(event, x, y, flags, param):
    global drawing, start_pt, end_pt, draw_mode, area_in, area_out
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing, start_pt = True, (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end_pt = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing, end_pt = False, (x, y)
        if start_pt and end_pt:
            rect = [start_pt,
                    (end_pt[0], start_pt[1]),
                    end_pt,
                    (start_pt[0], end_pt[1])]
            if draw_mode == 'in':
                area_in = rect;  print("Set IN  region:", area_in)
            elif draw_mode == 'out':
                area_out = rect; print("Set OUT region:", area_out)
        start_pt = end_pt = draw_mode = None

# ─────────────────────────── 5. HELPERS ────────────────────────────────────
def smooth(frame, k=3):
    return cv2.GaussianBlur(cv2.medianBlur(frame, k), (k, k), 1.2)

# ─────────────────────────── 6. PER‑FRAME LOGIC ────────────────────────────
def process_frame(frame, tracker, area_in, area_out,
                  states, entered, exited):
    # ----- foreground motion mask ------------------------------------------
    motion = bg_subtractor.apply(frame)
    _, motion = cv2.threshold(motion, 254, 255, cv2.THRESH_BINARY)
    motion = cv2.morphologyEx(
        motion, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 2
    )

    # ----- SSD‑300 inference -----------------------------------------------
    with torch.no_grad():
        inp = tfm(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        preds = ssd_model(inp)[0]

    boxes  = preds['boxes'].cpu().numpy().astype(int)
    labels = preds['labels'].cpu().numpy()
    scores = preds['scores'].cpu().numpy()

    # keep person (label==1) + score ≥ 0.30  (lowers threshold for upper‑body)
    dets = []
    for (x1, y1, x2, y2), lbl, scr in zip(boxes, labels, scores):
        if lbl != 1 or scr < 0.30:
            continue
        # optional motion filter: require ≥ 5 % of bbox pixels be “moving”
        roi = motion[y1:y2, x1:x2]
        if roi.size and cv2.countNonZero(roi) < 0.05 * roi.size:
            continue
        dets.append([x1, y1, x2, y2])
    dets = np.asarray(dets)

    # ----- SORT tracking ----------------------------------------------------
    tracked = tracker.update(dets) if dets.size else np.empty((0, 5))

    # ----- region masks -----------------------------------------------------
    h, w = frame.shape[:2]
    m_in  = np.zeros((h, w), np.uint8)
    m_out = np.zeros((h, w), np.uint8)
    if len(area_in)  == 4: cv2.fillPoly(m_in,  [np.int32(area_in)],  255)
    if len(area_out) == 4: cv2.fillPoly(m_out, [np.int32(area_out)], 255)

    overlay = frame.copy()
    alpha   = 0.5

    for tk in tracked:
        arr = tk.astype(int)
        x1, y1, x2, y2 = arr[:4]
        tid = arr[-1]                      # last column = ID (works 5/6 cols)

        # entry/exit logic via mask overlap
        in_px  = cv2.countNonZero(m_in [y1:y2, x1:x2])
        out_px = cv2.countNonZero(m_out[y1:y2, x1:x2])

        prev = states.get(tid)
        if prev is None:
            if in_px:  states[tid] = 'in'
            elif out_px: states[tid] = 'out'
        else:
            if prev == 'in' and out_px:
                exited.add(tid);  states.pop(tid)
            elif prev == 'out' and in_px:
                entered.add(tid); states.pop(tid)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cvzone.putTextRect(frame, f'ID {tid}', (x1, y1 - 10), 1, 1)

    blended = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return blended, len(entered), len(exited)

# ─────────────────────────── 7. MAIN LOOP ──────────────────────────────────
def main():
    global draw_mode, drawing_region, start_point, end_point, headcount_limit, alert_active, last_alert_time
    states, entered, exited = {}, set(), set()

    cv2.namedWindow('people_counter')
    cv2.setMouseCallback('people_counter', draw_rect)
    print("Press 'i' (IN), 'o' (OUT), 'l' (set Limit), ESC to exit.")

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1020, 500))
        frame = smooth(frame)

        # show rectangle while drawing + stored polygons
        if start_pt and end_pt:
            cv2.rectangle(frame, start_pt, end_pt, (0, 255, 255), 2)
        if len(area_in) == 4:
            cv2.polylines(frame, [np.int32(area_in)],  True, (0, 255, 0), 2)
        if len(area_out) == 4:
            cv2.polylines(frame, [np.int32(area_out)], True, (0, 0, 255), 2)

        out_frame, n_in, n_out = process_frame(
            frame, tracker, area_in, area_out,
            states, entered, exited
        )

        cv2.putText(out_frame, f'In : {n_in}',  (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(out_frame, f'Out: {n_out}', (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(out_frame, f'Headcount in event area: {n_in - n_out}', (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)


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
        k = cv2.waitKey(1) & 0xFF
        if k == 27:                    # ESC
            break
        elif k == ord('i'):
            draw_mode = 'in';  print("Draw IN region.")
        elif k == ord('o'):
            draw_mode = 'out'; print("Draw OUT region.")

        elif k==ord('l'):
            try:
                new_limit = int(input("Enter new headcount limit: "))
                if new_limit >= 0:
                    headcount_limit = new_limit
                    print(f"Headcount limit set to {headcount_limit}")
                        # Check if headcount exceeds limit
                    current_time = time.time()
                    if (n_in-n_out) > headcount_limit:
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

# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
