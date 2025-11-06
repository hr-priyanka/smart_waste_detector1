# improved_live_waste_with_training.py
import os
import time
import cv2
import numpy as np
from pathlib import Path

# Optional TTS
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

# ---- Feature helpers ----
def get_largest_object_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    contours_info = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    h,w = gray.shape
    if area < (0.005 * w * h):   # smaller threshold to capture smaller objects
        return None, None
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    return mask, largest

def lbp_image(gray, P=8, R=1):
    # basic LBP (uniform variants not used — keep small & robust)
    h,w = gray.shape
    out = np.zeros_like(gray, dtype=np.uint8)
    angles = [(np.cos(2*np.pi*i/P)*R, np.sin(2*np.pi*i/P)*R) for i in range(P)]
    for y in range(R, h-R):
        for x in range(R, w-R):
            center = gray[y,x]
            code = 0
            for i,(dx,dy) in enumerate(angles):
                xi = int(round(x + dx))
                yi = int(round(y + dy))
                code = (code << 1) | (1 if gray[yi,xi] >= center else 0)
            out[y,x] = code
    return out

def extract_features(roi, mask):
    """Return feature vector (numpy array) and dict for debug."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mean = cv2.mean(hsv, mask=mask)
    mean_h, mean_s, mean_v = mean[0], mean[1], mean[2]

    mask_area = cv2.countNonZero(mask)
    if mask_area == 0:
        return None, None

    # Specular highlights (very bright pixels)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, spec_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    spec_in_mask = cv2.bitwise_and(spec_mask, spec_mask, mask=mask)
    spec_count = cv2.countNonZero(spec_in_mask)
    specular_ratio = spec_count / mask_area

    # Edge density
    edges = cv2.Canny(gray, 60, 120)
    edges_in_mask = cv2.bitwise_and(edges, edges, mask=mask)
    edge_count = cv2.countNonZero(edges_in_mask)
    edge_ratio = edge_count / mask_area

    # Texture std
    _, stddev = cv2.meanStdDev(gray, mask=mask)
    texture_std = float(stddev[0][0])

    # Hu moments (shape)
    moments = cv2.moments(mask)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)  # scale

    # Color histogram (HSV hue bins inside mask)
    hue = hsv[:,:,0]
    hist = cv2.calcHist([hue], [0], mask, [16], [0,180]).flatten()
    hist = hist / (hist.sum() + 1e-12)

    # LBP histogram
    lbp = lbp_image(gray)
    lbp_in_mask = lbp[mask==255]
    if lbp_in_mask.size == 0:
        lbp_hist = np.zeros(16)
    else:
        lbp_hist, _ = np.histogram(lbp_in_mask, bins=16, range=(0,256))
        lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-12)

    # Consolidate into vector
    feat = np.hstack([
        mean_h/180.0, mean_s/255.0, mean_v/255.0,
        specular_ratio, edge_ratio, texture_std/255.0,
        hu_log, hist, lbp_hist, mask_area/ (roi.shape[0]*roi.shape[1])
    ]).astype(np.float32)

    debug = {
        "mean_h":mean_h, "mean_s":mean_s, "mean_v":mean_v,
        "specular_ratio":specular_ratio, "edge_ratio":edge_ratio,
        "texture_std":texture_std, "hu":hu_log.tolist()
    }
    return feat, debug

# ---- Dataset saving & training ----
DATA_DIR = Path("waste_dataset")
DATA_DIR.mkdir(exist_ok=True)
MODEL_PATH = Path("waste_model.joblib")

def save_sample(roi, mask, label_name):
    ts = int(time.time()*1000)
    roi_path = DATA_DIR / f"{ts}_{label_name}.jpg"
    mask_path = DATA_DIR / f"{ts}_{label_name}_mask.png"
    cv2.imwrite(str(roi_path), roi)
    cv2.imwrite(str(mask_path), mask)
    print("Saved sample:", roi_path.name, mask_path.name)

def load_dataset_and_features():
    files = sorted(DATA_DIR.glob("*_*.jpg"))
    X = []
    y = []
    for f in files:
        name = f.stem  # timestamp_label
        parts = name.split("_")
        if len(parts) < 2: continue
        label = parts[-1]
        mask_file = DATA_DIR / f"{f.stem}_{label}_mask.png"
        # Some earlier naming variants:
        alt_mask = DATA_DIR / f"{f.stem}_mask.png"
        if mask_file.exists():
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        elif alt_mask.exists():
            mask = cv2.imread(str(alt_mask), cv2.IMREAD_GRAYSCALE)
        else:
            # try to regenerate mask from roi
            roi = cv2.imread(str(f))
            mask, _ = get_largest_object_mask(roi)
            if mask is None:
                continue
        roi = cv2.imread(str(f))
        feat, _ = extract_features(roi, mask)
        if feat is None: continue
        X.append(feat)
        y.append(label)
    return np.array(X), np.array(y)

def train_and_save_model():
    try:
        from sklearn.svm import SVC
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        import joblib
    except Exception as e:
        print("scikit-learn and joblib are required to train. Install with: pip install scikit-learn joblib")
        return False

    X, y = load_dataset_and_features()
    if len(X) < 10:
        print("Not enough samples to train. Collect more (at least ~10-20 per class if possible). Found:", len(X))
        return False

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", probability=True))
    ])
    print("Training on", X.shape, "samples...")
    pipeline.fit(X, y)
    joblib.dump(pipeline, str(MODEL_PATH))
    print("Saved model to", MODEL_PATH)
    return True

def load_model_if_exists():
    try:
        import joblib
        if MODEL_PATH.exists():
            model = joblib.load(str(MODEL_PATH))
            return model
    except Exception:
        pass
    return None

# ---- Heuristic fallback (improved) ----
def heuristic_classify(debug_f):
    # improved combination of rules (still fallback)
    mean_s = debug_f["mean_s"]/255.0
    mean_v = debug_f["mean_v"]/255.0
    spec = debug_f["specular_ratio"]
    edge = debug_f["edge_ratio"]
    tex = debug_f["texture_std"]/100.0

    # rules
    if (spec > 0.02 and mean_s < 80) or (edge > 0.08 and mean_s < 70):
        return "Metal", {"Metal":0.9}
    if mean_v > 200 and mean_s < 60:
        return "Paper", {"Paper":0.8}
    if mean_s > 60:
        return "Plastic", {"Plastic":0.85}
    # default
    return "Unknown", {"Unknown":0.5}

# ---- Main loop ----
LABEL_KEYS = {ord('1'):"Plastic", ord('2'):"Paper", ord('3'):"Metal"}

def main():
    model = load_model_if_exists()
    if model:
        print("Loaded trained model.")
    else:
        print("No trained model found. Press 1/2/3 to save labeled samples, then press 't' to train.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Can't open camera. Change index or check camera.")
        return

    last_label = None
    last_time = 0
    print("Place object in center. Keys: 1=Plastic 2=Paper 3=Metal (save sample). 's' save unlabeled roi. 't' train. 'q' quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (800,600))
        h,w = frame.shape[:2]
        rw, rh = int(w*0.5), int(h*0.6)
        cx, cy = w//2, h//2
        x1,y1 = cx - rw//2, cy - rh//2
        x2,y2 = cx + rw//2, cy + rh//2
        roi = frame[y1:y2, x1:x2].copy()

        mask, contour = get_largest_object_mask(roi)
        display_text = "No object"
        debug = None
        conf_text = ""

        if mask is not None:
            feat, debug = extract_features(roi, mask)
            if feat is not None:
                if model is not None:
                    try:
                        probs = model.predict_proba(feat.reshape(1,-1))[0]
                        labels = model.classes_
                        best_idx = np.argmax(probs)
                        label = labels[best_idx]
                        conf = probs[best_idx]
                        display_text = f"{label}"
                        conf_text = f"{conf:.2f}"
                        # speak on change
                        if label != last_label and time.time() - last_time > 1.5:
                            speak(label if label!="Unknown" else "Unknown item")
                            last_label = label
                            last_time = time.time()
                    except Exception:
                        label, scores = heuristic_classify(debug)
                        display_text = label
                else:
                    label, scores = heuristic_classify(debug)
                    display_text = label

                # draw bounding rect of detected contour
                x,y,wc,hc = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x1+x, y1+y), (x1+x+wc, y1+y+hc), (0,255,0), 2)

        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,215,0), 2)
        cv2.putText(frame, f"Detected: {display_text} {conf_text}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        if debug:
            dbg_lines = [
                f"H={debug['mean_h']:.0f} S={debug['mean_s']:.0f} V={debug['mean_v']:.0f}",
                f"spec={debug['specular_ratio']:.3f} edge={debug['edge_ratio']:.3f} std={debug['texture_std']:.1f}"
            ]
            for i,line in enumerate(dbg_lines):
                cv2.putText(frame, line, (10,60+i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.imshow("Improved Live Waste Detection (training-enabled)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # save unlabeled ROI
        if key == ord('s') and mask is not None:
            ts = int(time.time()*1000)
            cv2.imwrite(str(DATA_DIR / f"{ts}_unlabeled.jpg"), roi)
            cv2.imwrite(str(DATA_DIR / f"{ts}_unlabeled_mask.png"), mask)
            print("Saved unlabeled sample.")
        # labeled saves
        if key in LABEL_KEYS and mask is not None:
            save_sample(roi, mask, LABEL_KEYS[key])
        if key == ord('t'):
            ok = train_and_save_model()
            if ok:
                model = load_model_if_exists()
        # reload model with 'r'
        if key == ord('r'):
            model = load_model_if_exists()
            print("Model reloaded." if model else "No model found.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
