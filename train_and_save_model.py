from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import ParameterGrid
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "egg_damage_pipeline.joblib"

IMAGE_SIZE = (64, 64)
SUBSETS = ["train", "validation", "test"]


def load_image(path, size=IMAGE_SIZE):
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def extract_all_features(img_rgb):
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
    except Exception:
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
        area = 0.0
        perimeter = 0.0
        w = 0.0
        h = 0.0
        aspect_ratio = 0.0
        extent = 0.0
        solidity = 0.0
        equivalent_diameter = 0.0

    features["shape_area"] = float(area)
    features["shape_perimeter"] = float(perimeter)
    features["shape_width"] = float(w)
    features["shape_height"] = float(h)
    features["shape_aspect_ratio"] = float(aspect_ratio)
    features["shape_extent"] = float(extent)
    features["shape_solidity"] = float(solidity)
    features["shape_equivalent_diameter"] = float(equivalent_diameter)

    return features


def build_subset_dataframe(subset_name):
    subset_path = DATASET_DIR / subset_name
    rows = []

    if not subset_path.exists():
        raise FileNotFoundError(f"Missing folder: {subset_path}")

    image_paths = (
        list(subset_path.rglob("*.jpg"))
        + list(subset_path.rglob("*.jpeg"))
        + list(subset_path.rglob("*.png"))
    )

    if not image_paths:
        raise ValueError(f"No images found in {subset_path}")

    for i, image_path in enumerate(image_paths, 1):
        print(f"[{subset_name}] Processing {i}/{len(image_paths)}: {image_path.name}")

        label = image_path.parent.name
        image_id = image_path.name

        img = load_image(image_path)
        if img is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        row = {"image_id": image_id, "label": label}
        row.update(extract_all_features(img))
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"No valid images were processed in {subset_path}")

    print(f"{subset_name} set shape: {df.shape}")
    return df


def evaluate(model, X, y):
    predictions = model.predict(X)
    score = accuracy_score(y, predictions)
    return score, predictions


def main():
    datasets = {}
    for subset in SUBSETS:
        print(f"\nLoading subset: {subset}")
        datasets[subset] = build_subset_dataframe(subset)

    print("\nLoaded datasets:")
    for subset in SUBSETS:
        print(f"{subset}: {type(datasets[subset])}, shape={datasets[subset].shape}")

    feature_names = [c for c in datasets["train"].columns if c not in ["image_id", "label"]]

    X_train = datasets["train"][feature_names]
    y_train_raw = datasets["train"]["label"]

    X_val = datasets["validation"][feature_names]
    y_val_raw = datasets["validation"]["label"]

    X_test = datasets["test"][feature_names]
    y_test_raw = datasets["test"]["label"]

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_val = label_encoder.transform(y_val_raw)
    y_test = label_encoder.transform(y_test_raw)

    k_values = sorted(set([5, 10, min(15, len(feature_names))]))

    model_configs = {
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {
                "selector__k": k_values,
                "model__max_depth": [3, 5, 8, None],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {
                "selector__k": k_values,
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 5, 10],
            },
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                "selector__k": k_values,
                "model__n_neighbors": [3, 5, 7],
            },
        ),
        "SVM": (
            SVC(probability=True, random_state=42),
            {
                "selector__k": k_values,
                "model__C": [0.1, 1, 10],
                "model__kernel": ["linear", "rbf"],
            },
        ),
        "Naive Bayes": (
            GaussianNB(),
            {
                "selector__k": k_values,
                "model__var_smoothing": [1e-9, 1e-8, 1e-7],
            },
        ),
    }

    best_name = None
    best_score = -1.0
    best_pipeline = None
    best_params = None

    for model_name, (base_model, grid) in model_configs.items():
        print(f"\nTraining {model_name}...")

        for params in ParameterGrid(grid):
            print(f"Testing params: {params}")

            fresh_model = clone(base_model)

            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("selector", SelectKBest(score_func=f_classif)),
                    ("model", fresh_model),
                ]
            )

            pipeline.set_params(**params)
            pipeline.fit(X_train, y_train)
            val_score, _ = evaluate(pipeline, X_val, y_val)

            print(f"Validation accuracy: {val_score:.4f}")

            if val_score > best_score:
                best_name = model_name
                best_score = val_score
                best_pipeline = pipeline
                best_params = params.copy()

    if best_pipeline is None:
        raise RuntimeError("No model was successfully trained.")

    test_score, test_predictions = evaluate(best_pipeline, X_test, y_test)
    selected_mask = best_pipeline.named_steps["selector"].get_support()
    selected_feature_names = [
        name for name, keep in zip(feature_names, selected_mask) if keep
    ]

    print("\nBest overall model")
    print(f"Best model: {best_name}")
    print(f"Validation accuracy: {best_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    print(f"Best params: {best_params}")
    print(f"Selected features: {selected_feature_names}")
    print("\nClassification report on test set:\n")
    print(classification_report(y_test, test_predictions, target_names=label_encoder.classes_))

    bundle = {
        "pipeline": best_pipeline,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
        "selected_feature_names": selected_feature_names,
        "best_model_name": best_name,
        "best_params": best_params,
        "validation_accuracy": best_score,
        "test_accuracy": test_score,
        "image_size": IMAGE_SIZE,
    }

    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved pipeline to: {MODEL_PATH}")


if __name__ == "__main__":
    main()