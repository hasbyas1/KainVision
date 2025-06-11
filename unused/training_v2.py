import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import cv2
from skimage import feature
from skimage.filters import gabor
from skimage.color import rgb2gray
import seaborn as sns
import joblib
from collections import Counter
import sys

# Path dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(base_dir, "train")
test_path = os.path.join(base_dir, "test")
output_dir = os.path.join(base_dir, "output")

print("Path to train dataset:", train_path)
print("Path to test dataset:", test_path)
print("Path to output directory:", output_dir)

# Konstanta
IMG_SIZE = 224
RANDOM_STATE = 42

# Kelas untuk ekstraksi fitur tekstur
class TextureFeatureExtractor:
    """Ekstraksi fitur tekstur untuk klasifikasi kain"""
    def __init__(self):
        pass

    def extract_glcm_features(self, image):
        gray = rgb2gray(image) if len(image.shape) == 3 else image
        gray = (gray * 255).astype(np.uint8)
        distances = [1, 2, 3]
        angles = [0, 45, 90, 135]
        properties = ['contrast', 'correlation', 'homogeneity', 'energy']
        features = []
        for distance in distances:
            for angle in angles:
                glcm = feature.graycomatrix(gray, [distance], [np.radians(angle)],
                                            levels=256, symmetric=True, normed=True)
                for prop in properties:
                    features.append(feature.graycoprops(glcm, prop)[0, 0])
        return np.array(features)

    def extract_lbp_features(self, image):
        gray = rgb2gray(image) if len(image.shape) == 3 else image
        gray = (gray * 255).astype(np.uint8)
        radius = 3
        n_points = 8 * radius
        lbp = feature.local_binary_pattern(gray, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2,
                               range=(0, n_points + 2), density=True)
        return hist

    def extract_gabor_features(self, image):
        gray = rgb2gray(image) if len(image.shape) == 3 else image
        features = []
        frequencies = [0.1, 0.3, 0.5]
        angles = [0, 45, 90, 135]
        for freq in frequencies:
            for angle in angles:
                filt_real, _ = gabor(gray, frequency=freq, theta=np.radians(angle))
                features.extend([
                    np.mean(filt_real),
                    np.var(filt_real),
                    np.std(filt_real)
                ])
        return np.array(features)

    def extract_color_features(self, image):
        if len(image.shape) == 3:
            features = []
            for channel in range(3):
                channel_data = image[:, :, channel].flatten()
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.mean((channel_data - np.mean(channel_data)) ** 3)
                ])
            for channel in range(3):
                hist, _ = np.histogram(image[:, :, channel], bins=32, range=(0, 1))
                features.extend(hist / np.sum(hist))
            return np.array(features)
        else:
            gray_data = image.flatten()
            features = [
                np.mean(gray_data),
                np.std(gray_data),
                np.mean((gray_data - np.mean(gray_data)) ** 3)
            ]
            hist, _ = np.histogram(image, bins=32, range=(0, 1))
            features.extend(hist / np.sum(hist))
            return np.array(features)

    def extract_all_features(self, image):
        glcm_features = self.extract_glcm_features(image)
        lbp_features = self.extract_lbp_features(image)
        gabor_features = self.extract_gabor_features(image)
        color_features = self.extract_color_features(image)
        return np.concatenate([glcm_features, lbp_features, gabor_features, color_features])

# Fungsi untuk memuat dan memproses data
def load_and_process_data(data_dir, feature_extractor, method_name):
    print(f"\nMemproses data dengan metode: {method_name}")
    X = []
    y = []
    class_names = os.listdir(data_dir)
    class_names = [cls for cls in class_names if os.path.isdir(os.path.join(data_dir, cls))]
    print(f"Kelas yang ditemukan: {class_names}")

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        image_files = [f for f in os.listdir(class_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Memproses kelas {class_name}: {len(image_files)} gambar")

        for i, img_file in enumerate(image_files, 1):
            img_path = os.path.join(class_dir, img_file)
            if i % 50 == 0 or i == 1:
                print(f"Memproses gambar {i}/{len(image_files)} di kelas {class_name}")
            try:
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Gagal membaca gambar: {img_path}")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                image = image.astype(np.float32) / 255.0
                if method_name == "Texture Features":
                    features = feature_extractor.extract_all_features(image)
                X.append(features)
                y.append(class_name)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

    print(f"Total gambar yang diproses: {len(X)}, Distribusi kelas: {Counter(y)}")
    return np.array(X), np.array(y), class_names

# Fungsi untuk melatih model
def train_and_evaluate_models(X, y, class_names, method_name):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'SVM': SVC(kernel='rbf', random_state=RANDOM_STATE),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }

    # Buat direktori TEST1 jika belum ada
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    print(f"\n=== Pelatihan Model dengan {method_name} ===")
    for model_name, model in models.items():
        print(f"\nMelatih {model_name}...")
        # Validasi silang untuk deteksi overfitting
        cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=5, scoring='accuracy')
        print(f"Cross-validation scores for {model_name}: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        model.fit(X_scaled, y_encoded)
        # Simpan model di TEST1
        model_path = os.path.join(output_dir, f"{model_name.replace(' ', '_')}_model.pkl")
        joblib.dump(model, model_path)
        print(f"Model disimpan di: {model_path}")

    # Simpan scaler dan label encoder di TEST1
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    le_path = os.path.join(output_dir, "label_encoder.pkl")
    joblib.dump(scaler, scaler_path)
    joblib.dump(le, le_path)
    print(f"Scaler disimpan di: {scaler_path}")
    print(f"Label encoder disimpan di: {le_path}")

    return results, scaler, le

# Fungsi untuk evaluasi pada dataset test
def evaluate_test_set(test_dir, feature_extractor, models, scaler, le, method_name):
    print(f"\n=== Evaluasi pada Dataset Test dengan {method_name} ===")
    X_test, y_test, class_names = load_and_process_data(test_dir, feature_extractor, method_name)
    if len(X_test) == 0:
        print("Error: Tidak ada data test yang diproses!")
        return

    X_test_scaled = scaler.transform(X_test)
    y_test_encoded = le.transform(y_test)

    for model_name, model in models.items():
        print(f"\nEvaluasi {model_name} pada dataset test...")
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test_encoded, y_pred)
        print(f"{model_name} Test Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report for {model_name} (Test Set):")
        print(classification_report(y_test_encoded, y_pred, target_names=class_names))
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test_encoded, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name} (Test Set) with {method_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()

# Main execution
def main():
    # Periksa direktori train
    if not os.path.exists(train_path):
        print(f"Error: Direktori {train_path} tidak ditemukan!")
        return

    print("Struktur direktori train:")
    for root, dirs, files in os.walk(train_path, topdown=True):
        print(f"Direktori: {root}")
        for d in dirs:
            print(f"  +-- {d} ({len(os.listdir(os.path.join(root, d)))} files)")
        break

    try:
        print("\n" + "=" * 50)
        print("METODE 1: TEXTURE FEATURES (GLCM + LBP + Gabor + Color)")
        print("=" * 50)

        texture_extractor = TextureFeatureExtractor()
        X_texture, y_texture, class_names = load_and_process_data(
            train_path, texture_extractor, "Texture Features"
        )

        if len(X_texture) > 0:
            print(f"Total gambar: {len(X_texture)}, Distribusi kelas: {Counter(y_texture)}")
            results_texture, scaler_texture, le_texture = train_and_evaluate_models(
                X_texture, y_texture, class_names, "Texture Features"
            )

            # Evaluasi pada dataset test
            if os.path.exists(test_path):
                models = {
                    'Random Forest': joblib.load(os.path.join(output_dir, "Random_Forest_model.pkl")),
                    'SVM': joblib.load(os.path.join(output_dir, "SVM_model.pkl")),
                    'K-Nearest Neighbors': joblib.load(os.path.join(output_dir, "KNeighbors_model.pkl"))
                }
                evaluate_test_set(test_path, texture_extractor, models, scaler_texture, le_texture, "Texture Features")
            else:
                print(f"Direktori test {test_path} tidak ditemukan. Lewati evaluasi test.")

        else:
            print("Error: Tidak ada data yang diproses. Periksa struktur dataset.")

    except Exception as e:
        print(f"Error: {e}")
        print("\nPastikan dataset memiliki struktur direktori yang benar:")
        print("train/")
        print("├── 001/")
        print("│   ├── img1.png")
        print("│   └── ...")
        print("├── 002/")
        print("└── ...")

if __name__ == "__main__":
    main()