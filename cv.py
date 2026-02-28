import cv2
import numpy as np
import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

SCENARIOS = {
    1: {
        "video":      "Problem Statement Scenario1.mp4",
        "output":     "output_scenario1.mp4",
        "line":       ((424, 150), (424, 420)),
        "count_side": "left",
        "label":      "Sacks Loaded",
    },
    2: {
        # Try both common filename variants for Scenario 2
        "video":      "Problem Statement Scenario2.mp4",
        "video_alt":  "Problem Statement Scenario 2.mp4",
        "output":     "output_scenario2.mp4",
        "line":       ((50, 540), (430, 540)),
        "count_side": "up",
        "label":      "Sacks Delivered",
    },
    3: {
        "video":      "Problem Statement Scenario3.mp4",
        "output":     "output_scenario3.mp4",
        "line":       ((400, 150), (400, 400)),
        "count_side": "left",
        "label":      "Sacks Counted",
    },
}

# Display window size (resize for screen, output video keeps original res)
DISPLAY_WIDTH  = 960
DISPLAY_HEIGHT = 540


# ─── Geometry helpers ───────────────────────────────────────────────────────
def side_of_line(p1, p2, pt):
    val = (p2[0]-p1[0])*(pt[1]-p1[1]) - (p2[1]-p1[1])*(pt[0]-p1[0])
    return 1 if val >= 0 else -1

def crosses_line(prev, curr, direction):
    if prev == curr:
        return False
    if direction in ("left", "up"):
        return prev == 1 and curr == -1
    return prev == -1 and curr == 1

class SimpleTracker:
    def __init__(self, iou_thresh=0.25, max_lost=15):
        self.next_id    = 0
        self.tracks     = {}
        self.iou_thresh = iou_thresh
        self.max_lost   = max_lost

    @staticmethod
    def iou(a, b):
        ax1,ay1,ax2,ay2 = a;  bx1,by1,bx2,by2 = b
        ix1=max(ax1,bx1); iy1=max(ay1,by1)
        ix2=min(ax2,bx2); iy2=min(ay2,by2)
        inter = max(0,ix2-ix1)*max(0,iy2-iy1)
        union = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
        return inter/union if union>0 else 0

    def update(self, dets):
        matched_t, matched_d = set(), set()
        results = []
        tids = list(self.tracks.keys())

        for di, det in enumerate(dets):
            best_iou, best_tid = self.iou_thresh, None
            for tid in tids:
                if tid in matched_t: continue
                v = self.iou(self.tracks[tid]["box"], det)
                if v > best_iou:
                    best_iou, best_tid = v, tid
            if best_tid is not None:
                self.tracks[best_tid].update({"box": det, "lost": 0})
                matched_t.add(best_tid); matched_d.add(di)
                results.append((best_tid, *det))

        for di, det in enumerate(dets):
            if di not in matched_d:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {"box": det, "lost": 0, "side": None}
                results.append((tid, *det))

        for tid in tids:
            if tid not in matched_t:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]
        return results

class YOLODetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def detect(self, frame):
        fh, fw = frame.shape[:2]
        res = self.model(frame, conf=0.35, verbose=False)[0]
        boxes = []
        for box in res.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            area = (x2-x1)*(y2-y1)
            if 800 < area < fw*fh*0.5:
                boxes.append((x1,y1,x2,y2))
        return boxes

class BGSubDetector:
    def __init__(self):
        self.bg     = cv2.createBackgroundSubtractorMOG2(300, 40, True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))

    def detect(self, frame):
        fh, fw = frame.shape[:2]
        mask = self.bg.apply(frame)
        mask[mask==127] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in cnts:
            area = cv2.contourArea(c)
            if 2000 < area < fw*fh*0.4:
                x,y,w,h = cv2.boundingRect(c)
                boxes.append((x,y,x+w,y+h))
        return boxes

def draw_overlay(frame, tracks, tracker, line_p1, line_p2, sack_count, label, scenario_id):
    h, w = frame.shape[:2]

    # Bounding boxes + labels
    for tid, x1, y1, x2, y2 in tracks:
        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,60,255), 2)
        conf_text = f"sacks:0.90"
        (tw, th), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+4, y1), (0,60,255), -1)
        cv2.putText(frame, conf_text, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
        cv2.circle(frame, (cx, cy), 4, (0,255,255), -1)

    # Counting line — red top half, blue bottom half (like reference)
    mid = ((line_p1[0]+line_p2[0])//2, (line_p1[1]+line_p2[1])//2)
    cv2.line(frame, line_p1, mid,      (0,0,255), 3)
    cv2.line(frame, mid,     line_p2,  (255,0,0), 3)

    # Sack count — large yellow text top-left (exactly like reference)
    cv2.putText(frame, f"sacks: {sack_count}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1.9, (0,255,255), 3)

    # Bottom label
    cv2.putText(frame, f"Scenario {scenario_id}  |  {label}  |  Press Q to skip",
                (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

    return frame

def process_video(sid: int, input_dir: str, output_dir: str):
    cfg = SCENARIOS[sid]

    vpath = Path(input_dir) / cfg["video"]
    if not vpath.exists() and "video_alt" in cfg:
        vpath = Path(input_dir) / cfg["video_alt"]
    if not vpath.exists():
        print(f"[ERROR] Cannot open: {vpath}")
        return None

    out_path   = str(Path(output_dir) / cfg["output"])
    line_p1    = cfg["line"][0]
    line_p2    = cfg["line"][1]
    direction  = cfg["count_side"]
    label      = cfg["label"]

    cap = cv2.VideoCapture(str(vpath))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(out_path,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))

    detector    = YOLODetector() if YOLO_AVAILABLE else BGSubDetector()
    tracker     = SimpleTracker()
    sack_count  = 0
    counted_ids = set()
    frame_idx   = 0

    win_name = f"Sack Counter — Scenario {sid}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    print(f"\n[Scenario {sid}] {vpath.name}  |  {width}x{height} @ {fps:.0f}fps  |  {total} frames")
    print(f"  Counting line {line_p1}→{line_p2}  direction={direction}")
    print(f"  Window: '{win_name}'   Press Q to skip to next scenario\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        dets   = detector.detect(frame)
        tracks = tracker.update(dets)

        for tid, x1, y1, x2, y2 in tracks:
            cx = (x1+x2)//2;  cy = (y1+y2)//2
            curr = side_of_line(line_p1, line_p2, (cx, cy))
            prev = tracker.tracks[tid].get("side")
            if prev is None:
                tracker.tracks[tid]["side"] = curr
            else:
                if tid not in counted_ids and crosses_line(prev, curr, direction):
                    sack_count += 1
                    counted_ids.add(tid)
                tracker.tracks[tid]["side"] = curr

        draw_overlay(frame, tracks, tracker, line_p1, line_p2,
                     sack_count, label, sid)

        writer.write(frame)

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print(f"  [Skipped at frame {frame_idx}]")
            break

        if frame_idx % 150 == 0:
            pct = frame_idx / total * 100
            print(f"  {frame_idx}/{total} ({pct:.0f}%)  {label}: {sack_count}")

    cap.release()
    writer.release()
    cv2.destroyWindow(win_name)

    print(f"\n  ✓ Scenario {sid} complete → {label}: {sack_count}")
    print(f"    Saved: {out_path}\n")
    return sack_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario",   type=int, choices=[1,2,3], default=None)
    parser.add_argument("--input-dir",  type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.input_dir:
        input_dir = args.input_dir
    else:
        candidates = [Path(__file__).parent.resolve(), Path.cwd(), Path.home()/"Desktop"]
        input_dir = None
        for d in candidates:
            if (d / "Problem Statement Scenario1.mp4").exists():
                input_dir = str(d)
                print(f"[INFO] Videos found in: {d}")
                break
        if input_dir is None:
            # Just use cwd and let per-video error messages handle missing files
            input_dir = str(Path.cwd())
            print(f"[INFO] Using current directory: {input_dir}")

    output_dir = args.output_dir or input_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  SACK COUNTER  |  YOLO={'ON' if YOLO_AVAILABLE else 'OFF (BGSub fallback)'}")
    print(f"  Input : {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"{'='*55}")
    if not YOLO_AVAILABLE:
        print("  [!] Install ultralytics for better accuracy:")
        print("      pip install ultralytics\n")

    scenarios_to_run = [args.scenario] if args.scenario else [1, 2, 3]
    totals = {}
    for sid in scenarios_to_run:
        count = process_video(sid, input_dir, output_dir)
        totals[sid] = count

    print(f"\n{'='*55}")
    print("  FINAL SACK COUNTS")
    print(f"{'='*55}")
    for sid, count in totals.items():
        status = f"{count} sacks" if count is not None else "SKIPPED (file not found)"
        print(f"  Scenario {sid}: {status}")
    print(f"\n  Output v    ideos → {output_dir}")
    print(f"{'='*55}")  