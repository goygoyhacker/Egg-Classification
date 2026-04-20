import streamlit as st
import joblib
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

# Load trained model
bundle = joblib.load("egg_damage_pipeline.joblib")
pipeline = bundle["pipeline"]
label_encoder = bundle["label_encoder"]
feature_names = bundle["feature_names"]
image_size = bundle["image_size"]

# Define which label means damaged
DAMAGED_LABELS = {"damaged"}  # CHANGE if your label is different


def extract_features(img_rgb):
    features = {}

    for i, c in enumerate(["R", "G", "B"]):
        channel = img_rgb[:, :, i]
        features[f"{c}_mean"] = float(np.mean(channel))
        features[f"{c}_std"] = float(np.std(channel))

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    features["gray_mean"] = float(np.mean(gray))
    features["gray_std"] = float(np.std(gray))

    glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
    features["glcm_contrast"] = float(graycoprops(glcm, "contrast")[0, 0])
    features["glcm_energy"] = float(graycoprops(glcm, "energy")[0, 0])
    features["glcm_correlation"] = float(graycoprops(glcm, "correlation")[0, 0])
    features["glcm_homogeneity"] = float(graycoprops(glcm, "homogeneity")[0, 0])

    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    features["lbp_mean"] = float(np.mean(lbp))
    features["lbp_std"] = float(np.std(lbp))

    try:
        hog_feat = hog(
            gray,
            orientations=6,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        features["hog_mean"] = float(np.mean(hog_feat))
        features["hog_std"] = float(np.std(hog_feat))
    except:
        features["hog_mean"] = 0.0
        features["hog_std"] = 0.0

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h else 0.0
        rect_area = w * h
        extent = area / rect_area if rect_area else 0.0

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area else 0.0

        equivalent_diameter = np.sqrt(4 * area / np.pi) if area else 0.0
    else:
        area = perimeter = w = h = 0.0
        aspect_ratio = extent = solidity = equivalent_diameter = 0.0

    features["shape_area"] = float(area)
    features["shape_perimeter"] = float(perimeter)
    features["shape_width"] = float(w)
    features["shape_height"] = float(h)
    features["shape_aspect_ratio"] = float(aspect_ratio)
    features["shape_extent"] = float(extent)
    features["shape_solidity"] = float(solidity)
    features["shape_equivalent_diameter"] = float(equivalent_diameter)

    return features


# UI
st.title("Egg Damage Detection")

uploaded_file = st.file_uploader("Upload an egg image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, image_size)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

    features = extract_features(img_rgb)

    X = np.array([[features[f] for f in feature_names]])

    pred = pipeline.predict(X)
    proba = pipeline.predict_proba(X)

    label = label_encoder.inverse_transform(pred)[0]
    confidence = np.max(proba) * 100

    st.write(f"Prediction: **{label}**")
    st.write(f"Confidence: **{confidence:.2f}%**")  
    
    # if label in DAMAGED_LABELS:
    #     st.error("Egg is DAMAGED")
    # else:
    #     st.success("Egg is NOT DAMAGED")