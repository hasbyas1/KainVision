import os
import cv2
import numpy as np
from skimage import feature
from skimage.filters import gabor
from skimage.color import rgb2gray
import joblib
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Konstanta
IMG_SIZE = 224

# Kelas untuk ekstraksi fitur (salin dari kode asli)
class TextureFeatureExtractor:
    """Ekstraksi fitur tekstur untuk klasifikasi kain"""
    def __init__(self):
        self.scaler = None

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

def predict_new_image(image_path, feature_extractor, model, scaler, le, method_name="Texture Features"):
    """Memprediksi kelas untuk gambar kain baru"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Gagal membaca gambar: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image.astype(np.float32) / 255.0
        if method_name == "Texture Features":
            features = feature_extractor.extract_all_features(image)
        features_scaled = scaler.transform([features])
        pred_encoded = model.predict(features_scaled)
        pred_label = le.inverse_transform(pred_encoded)[0]
        return pred_label
    except Exception as e:
        return f"Error: {e}"

class FabricClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Klasifikasi Kain")
        self.root.geometry("800x600")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, "TEST1")

        # Muat model, scaler, dan label encoder
        try:
            self.random_forest_model = joblib.load(os.path.join(model_dir, "Random_Forest_model.pkl"))
            self.svm_model = joblib.load(os.path.join(model_dir, "SVM_model.pkl"))
            self.knn_model = joblib.load(os.path.join(model_dir, "KNeighbors_model.pkl"))
            self.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
            self.le = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
        except FileNotFoundError as e:
            messagebox.showerror("Error", f"File model tidak ditemukan: {e}\nJalankan kode pelatihan terlebih dahulu!")
            self.root.quit()
            return

        self.feature_extractor = TextureFeatureExtractor()

        # Komponen GUI
        self.label = tk.Label(root, text="Pilih gambar kain untuk klasifikasi", font=("Arial", 14))
        self.label.pack(pady=10)

        self.btn_select = tk.Button(root, text="Pilih Gambar", command=self.select_image)
        self.btn_select.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Arial", 12), wraplength=700)
        self.result_label.pack(pady=10)

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if not file_path:
            return

        # Tampilkan gambar
        try:
            image = Image.open(file_path)
            image = image.resize((400, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Simpan referensi
        except Exception as e:
            messagebox.showerror("Error", f"Gagal memuat gambar: {e}")
            return

        # Prediksi
        predictions = {}
        for model_name, model in [
            ("Random Forest", self.random_forest_model),
            ("SVM", self.svm_model),
            ("K-Nearest Neighbors", self.knn_model)
        ]:
            predicted_class = predict_new_image(
                file_path, self.feature_extractor, model, self.scaler, self.le
            )
            predictions[model_name] = predicted_class

        # Tampilkan hasil
        result_text = "\n".join([f"{model_name}: {pred_class}" for model_name, pred_class in predictions.items()])
        self.result_label.config(text=result_text)

def main():
    root = tk.Tk()
    app = FabricClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()