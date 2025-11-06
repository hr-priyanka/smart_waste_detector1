# improved_live_waste.py
import cv2
import numpy as np
import time

# Optional text-to-speech
try:
    import pyttsx3
    tts_engine = pyttsx3.init()
except Exception:
    tts_engine = None

def speak(text):
    if tts_engine:
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception:
            pass

def get_largest_object_mask(img):
    """Return mask (255 where object) and contour of largest object in img (BGR)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    # Otsu threshold
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If threshold is mostly white (likely dark object on light bg) invert it
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours_info = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    h, w = gray.shape
    if area < (0.01 * w * h):   # too small -> ignore
        return None, None

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    return mask, largest

def compute_features(roi, mask):
    """Compute mean HSV, specular_ratio, edge_ratio, texture_std inside mask."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mean = cv2.mean(hsv, mask=mask)  # (h,s,v,a)
    mean_h, mean_s, mean_v = mean[0], mean[1], mean[2]

    # Mask area (number of pixels inside object)
    mask_area = cv2.countNonZero(mask)
    if mask_area == 0:
        return None

    # Specular highlights (very bright pixels) inside mask
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # threshold for very bright pixels (tuneable)
    _, spec_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    spec_in_mask = cv2.bitwise_and(spec_mask, spec_mask, mask=mask)
    spec_count = cv2.countNonZero(spec_in_mask)
    specular_ratio = spec_count / mask_area

    # Edge / texture density
    edges = cv2.Canny(gray, 60, 120)
    edges_in_mask = cv2.bitwise_and(edges, edges, mask=mask)
    edge_count = cv2.countNonZero(edges_in_mask)
    edge_ratio = edge_count / mask_area

    # Texture std (brightness/stddev)
    mean_g, stddev = cv2.meanStdDev(gray, mask=mask)
    texture_std = float(stddev[0][0])

    return {
        "mean_h": mean_h,
        "mean_s": mean_s,
        "mean_v": mean_v,
        "specular_ratio": specular_ratio,
        "edge_ratio": edge_ratio,
        "texture_std": texture_std,
        "mask_area": mask_area
    }

def classify_from_features(f):
    """Score each class and return best label + scores dict."""
    # Normalize values
    mean_s = f["mean_s"] / 255.0
    mean_v = f["mean_v"] / 255.0
    spec = f["specular_ratio"]   # already ratio
    edge = f["edge_ratio"]       # already ratio
    tex = f["texture_std"] / 100.0  # rough normalize

    # Heuristic scoring (weights can be tuned)
    plastic_score = mean_s                           # colored plastics tend to have higher saturation
    paper_score   = mean_v * (1 - mean_s)           # bright & low saturation
    metal_score   = (spec * 3.0) + (1 - mean_s) * 0.4 + (edge * 2.0)  # highlights + low saturation + edges

    scores = {
        "Plastic": float(plastic_score),
        "Paper": float(paper_score),
        "Metal": float(metal_score)
    }

    # Choose best
    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    # Require a minimum confidence to avoid random guesses
    if best_score < 0.25:
        return "Unknown", scores

    return best_label, scores

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera. Try changing camera index.")
        return

    last_label = None
    last_time = 0

    print("Place the object inside the center rectangle. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (800, 600))
        h, w = frame.shape[:2]
        # center ROI where user should place object (tweak sizes if needed)
        rw, rh = int(w * 0.5), int(h * 0.6)
        cx, cy = w // 2, h // 2
        x1, y1 = cx - rw//2, cy - rh//2
        x2, y2 = cx + rw//2, cy + rh//2

        roi = frame[y1:y2, x1:x2].copy()
        mask, contour = get_largest_object_mask(roi)

        display_text = "No object"
        debug_vals = None

        if mask is not None:
            f = compute_features(roi, mask)
            if f is not None:
                label, scores = classify_from_features(f)
                display_text = f"{label}"
                debug_vals = (f, scores)

                # Only speak when label changes and cooldown passed
                if label != last_label and time.time() - last_time > 1.5:
                    speak(label if label != "Unknown" else "Unknown item")
                    last_label = label
                    last_time = time.time()

                # draw detected contour on ROI (scale to full frame)
                # draw rectangle around detected object on preview
                x,y,wc,hc = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x1 + x, y1 + y), (x1 + x + wc, y1 + y + hc), (0,255,0), 2)

        # Draw ROI rectangle
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,215,0), 2)
        cv2.putText(frame, f"Detected: {display_text}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        # If debug info exists, show on screen
        if debug_vals:
            f, scores = debug_vals
            dbg_lines = [
                f"H={f['mean_h']:.0f} S={f['mean_s']:.0f} V={f['mean_v']:.0f}",
                f"spec={f['specular_ratio']:.3f} edge={f['edge_ratio']:.3f} std={f['texture_std']:.1f}",
                f"Scores P={scores['Plastic']:.2f} Pa={scores['Paper']:.2f} M={scores['Metal']:.2f}"
            ]
            for i, line in enumerate(dbg_lines):
                cv2.putText(frame, line, (10, 60 + i*25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.imshow("Improved Live Waste Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # optional: press 's' to save annotated ROI for dataset
        if key == ord('s') and mask is not None:
            timestamp = int(time.time())
            cv2.imwrite(f"capture_{timestamp}.jpg", roi)
            print("Saved capture:", f"capture_{timestamp}.jpg")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
