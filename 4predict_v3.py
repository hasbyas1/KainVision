import os
import cv2
import numpy as np
from skimage import feature
import joblib
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Directory configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, "4model")

# Load models and preprocessors at startup
try:
    RF_MODEL = joblib.load(os.path.join(model_dir, "RF_lbp_model.pkl"))
    SVM_MODEL = joblib.load(os.path.join(model_dir, "SVM_lbp_model.pkl"))
    KNN_MODEL = joblib.load(os.path.join(model_dir, "KNN_lbp_model.pkl"))
    SCALER = joblib.load(os.path.join(model_dir, "lbp_scaler.pkl"))
    LABEL_ENCODER = joblib.load(os.path.join(model_dir, "lbp_label_encoder.pkl"))
except FileNotFoundError as e:
    print(f"Error loading models: {e}")
    print("Make sure all model files exist in '4model' directory")

class LBPExtractor:
    """Simple LBP feature extraction"""
    
    def extract_lbp_features(self, image):
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Ensure uint8
        gray = (gray * 255).astype(np.uint8)
        
        # LBP parameters (same as training)
        radius = 3
        n_points = 24
        
        # Extract LBP
        lbp = feature.local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Calculate histogram
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2,
                             range=(0, n_points + 2), density=True)
        return hist

def predict_fabric(image_path, extractor, model, scaler, le):
    """Predict fabric class using LBP"""
    try:
        # Read and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            return "Error: Cannot read image"
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        
        # Extract LBP features
        features = extractor.extract_lbp_features(image)
        
        # Scale and predict
        features_scaled = scaler.transform([features])
        pred_encoded = model.predict(features_scaled)
        pred_label = le.inverse_transform(pred_encoded)[0]
        
        return pred_label
        
    except Exception as e:
        return f"Error: {e}"

class SimpleFabricGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LBP Fabric Classifier")
        self.root.geometry("900x800")  # Much larger window
        
        # Load models
        self.load_models()
        
        # Setup GUI
        self.setup_gui()
    
    def load_models(self):
        """Load LBP models and preprocessors"""        
        try:
            self.rf_model = RF_MODEL
            self.svm_model = SVM_MODEL
            self.knn_model = KNN_MODEL
            self.scaler = SCALER
            self.le = LABEL_ENCODER
            self.extractor = LBPExtractor()
            
        except NameError:
            messagebox.showerror("Error", "LBP models not found!\nMake sure all model files exist in '4model' directory.")
            self.root.quit()
    
    def setup_gui(self):
        """Setup simple GUI"""
        # Title
        tk.Label(self.root, text="LBP Fabric Classifier", 
                font=("Arial", 16, "bold")).pack(pady=10)
        
        # Select button
        tk.Button(self.root, text="Select Fabric Image", 
                 command=self.select_image,
                 font=("Arial", 12), bg="#4CAF50", fg="white").pack(pady=10)
        
        # Image display (remove fixed width/height constraints)
        self.image_label = tk.Label(self.root, text="No image selected", 
                                   bg="lightgray")
        self.image_label.pack(pady=10)
        
        # Results
        self.result_label = tk.Label(self.root, text="", font=("Arial", 12))
        self.result_label.pack(pady=10)
    
    def select_image(self):
        """Handle image selection and prediction"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp *.tiff")]
        )
        
        if not file_path:
            return
        
        # Display image (much larger, no thumbnail restriction)
        try:
            image = Image.open(file_path)
            
            # Resize to a specific large size instead of thumbnail
            image = image.resize((500, 500), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
        except:
            messagebox.showerror("Error", "Cannot display image")
            return
        
        # Predict with all models
        self.predict_all(file_path)
    
    def predict_all(self, image_path):
        """Predict using all three models"""
        models = [
            ("Random Forest", self.rf_model),
            ("SVM", self.svm_model),
            ("K-NN", self.knn_model)
        ]
        
        predictions = []
        for name, model in models:
            pred = predict_fabric(image_path, self.extractor, model, self.scaler, self.le)
            predictions.append(f"{name}: {pred}")
        
        # Check consensus
        pred_values = [p.split(": ")[1] for p in predictions if not p.startswith("Error")]
        if len(set(pred_values)) == 1 and pred_values:
            predictions.append(f"\n✓ Consensus: {pred_values[0]}")
        
        self.result_label.config(text="\n".join(predictions))

def main():
    root = tk.Tk()
    app = SimpleFabricGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()