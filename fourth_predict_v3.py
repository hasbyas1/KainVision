import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import feature
import joblib
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from first_preprocessing_v3 import FabricLBPPreprocessor
from datetime import datetime

# Directory configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, "4model")
step_output_dir = os.path.join(current_dir, "5output_step_by_step")
evaluation_dir = os.path.join(step_output_dir, "evaluations")

# Create directories
os.makedirs(step_output_dir, exist_ok=True)
os.makedirs(evaluation_dir, exist_ok=True)

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

class StepByStepPreprocessor(FabricLBPPreprocessor):
    """Even more efficient - use original preprocess_for_lbp method and capture intermediate steps"""
    
    def preprocess_for_lbp_with_steps(self, image, filename_prefix="preprocessing"):
        """
        Modified version of original preprocess_for_lbp that saves each step
        """
        print(f"Starting preprocessing for: {filename_prefix}")
        
        # Step 1: Save original image
        print("Step 1: Saving original image...")
        self._save_step(image, filename_prefix, "01_original")
        
        # Step 2: Resize (using parent method)
        print("Step 2: Resizing image...")
        processed = self.resize_image(image, scale_factor=0.9)
        self._save_step(processed, filename_prefix, "02_resized")
        
        # Step 3: Convert to grayscale
        print("Step 3: Converting to grayscale...")
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        self._save_step(processed, filename_prefix, "03_grayscale")
        
        # Step 4: Light Gaussian blur
        print("Step 4: Applying Gaussian blur...")
        processed = cv2.GaussianBlur(processed, (3, 3), 0.8)
        self._save_step(processed, filename_prefix, "04_blurred")
        
        # Step 5: Adaptive normalization (using parent method)
        print("Step 5: Applying adaptive normalization...")
        processed = self.adaptive_normalization(processed)
        self._save_step(processed, filename_prefix, "05_normalized")
        
        # Step 6: Mild CLAHE contrast enhancement
        print("Step 6: Applying CLAHE contrast enhancement...")
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        processed = clahe.apply(processed)
        self._save_step(processed, filename_prefix, "06_final_contrast")
        
        print(f"All preprocessing steps saved in: {step_output_dir}")
        return processed
    
    def _save_step(self, image, filename_prefix, step_name):
        """Helper method to save preprocessing step"""
        file_path = os.path.join(step_output_dir, f"{filename_prefix}_{step_name}.png")
        if len(image.shape) == 3:
            cv2.imwrite(file_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(file_path, image)
        print(f"  - Saved: {filename_prefix}_{step_name}.png")

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
        return hist, lbp

class LBPEvaluator:
    """Class untuk mengevaluasi dan menyimpan hasil LBP"""
    
    def __init__(self, evaluation_dir):
        self.evaluation_dir = evaluation_dir
        os.makedirs(evaluation_dir, exist_ok=True)
    
    def save_lbp_evaluation(self, image_name, lbp_features, lbp_image, original_image, predictions):
        """Simpan evaluasi LBP dalam bentuk Excel dan histogram"""
        
        # 1. Simpan ke Excel
        self._save_to_excel(image_name, lbp_features, predictions)
        
        # 2. Buat histogram visualization
        self._create_histogram_visualization(image_name, lbp_features, lbp_image, original_image, predictions)
        
        # 3. TAMBAHAN: Simpan LBP pattern image saja
        self._save_lbp_pattern_only(image_name, lbp_image)
        
        print(f"LBP evaluation saved for: {image_name}")
    
    def _save_lbp_pattern_only(self, image_name, lbp_image):
        """Simpan hanya LBP pattern image dalam berbagai format"""
        
        # Format 1: Simple grayscale PNG
        simple_path = os.path.join(self.evaluation_dir, f"{image_name}_lbp_pattern.png")
        # Normalize LBP values to 0-255 range
        lbp_normalized = ((lbp_image - lbp_image.min()) / (lbp_image.max() - lbp_image.min()) * 255).astype(np.uint8)
        cv2.imwrite(simple_path, lbp_normalized)
        print(f"LBP pattern (simple) saved: {simple_path}")
        
        # Format 2: Styled dengan matplotlib dan colorbar
        styled_path = os.path.join(self.evaluation_dir, f"{image_name}_lbp_pattern_styled.png")
        plt.figure(figsize=(8, 8))
        im = plt.imshow(lbp_image, cmap='gray', interpolation='nearest')
        plt.title(f'LBP Pattern - {image_name}', fontweight='bold', fontsize=16)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(styled_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"LBP pattern (styled) saved: {styled_path}")
        
        # Format 3: Original vs LBP comparison
        comparison_path = os.path.join(self.evaluation_dir, f"{image_name}_original_vs_lbp.png")
        # This will be handled by _create_comparison method if needed
        
    def _save_to_excel(self, image_name, lbp_features, predictions):
        """Simpan features LBP ke Excel"""
        
        # Prepare data
        data = {
            'image_name': [image_name],
            'timestamp': [datetime.now().isoformat()],
            'predicted_class_RF': [predictions.get('Random Forest', 'N/A')],
            'predicted_class_SVM': [predictions.get('SVM', 'N/A')],
            'predicted_class_KNN': [predictions.get('K-NN', 'N/A')]
        }
        
        # Add LBP features
        for i, feature_val in enumerate(lbp_features):
            data[f'feature_{i:03d}'] = [round(feature_val, 6)]
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save to Excel
        excel_path = os.path.join(self.evaluation_dir, f"{image_name}_lbp_features.xlsx")
        
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='LBP_Features', index=False)
                
                # Add feature statistics sheet
                feature_cols = [col for col in df.columns if col.startswith('feature_')]
                feature_stats = df[feature_cols].T
                feature_stats.columns = ['Value']
                feature_stats['Bin_Index'] = range(len(feature_stats))
                feature_stats = feature_stats[['Bin_Index', 'Value']]
                feature_stats.to_excel(writer, sheet_name='Feature_Statistics', index=True)
            
            print(f"Excel saved: {excel_path}")
            
        except ImportError:
            # Fallback to CSV
            csv_path = excel_path.replace('.xlsx', '.csv')
            df.to_csv(csv_path, index=False)
            print(f"CSV saved (Excel not available): {csv_path}")
    
    def _create_histogram_visualization(self, image_name, lbp_features, lbp_image, original_image, predictions):
        """Buat visualisasi histogram LBP"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'LBP Analysis - {image_name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Original Image
        ax1 = axes[0, 0]
        if len(original_image.shape) == 3:
            ax1.imshow(original_image)
        else:
            ax1.imshow(original_image, cmap='gray')
        ax1.set_title('Original Image', fontweight='bold')
        ax1.axis('off')
        
        # Plot 2: LBP Image
        ax2 = axes[0, 1]
        im = ax2.imshow(lbp_image, cmap='gray')
        ax2.set_title('LBP Pattern Image', fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        
        # Plot 3: LBP Histogram
        ax3 = axes[1, 0]
        bars = ax3.bar(range(len(lbp_features)), lbp_features, 
                       color='skyblue', alpha=0.7, edgecolor='black')
        ax3.set_title('LBP Feature Histogram', fontweight='bold')
        ax3.set_xlabel('LBP Bin Index')
        ax3.set_ylabel('Frequency (Density)')
        ax3.grid(True, alpha=0.3)
        
        # Highlight top 5 features
        top_indices = np.argsort(lbp_features)[-5:]
        for idx in top_indices:
            bars[idx].set_color('red')
            bars[idx].set_alpha(0.8)
        
        # Add value labels on top bars
        for i, v in enumerate(lbp_features):
            if i in top_indices:
                ax3.text(i, v + 0.002, f'{v:.3f}', ha='center', va='bottom', 
                        fontsize=8, fontweight='bold')
        
        # Plot 4: Predictions Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create prediction text
        pred_text = "Model Predictions:\n" + "="*20 + "\n"
        for model, pred in predictions.items():
            pred_text += f"{model}: {pred}\n"
        
        # Add feature statistics
        pred_text += f"\nFeature Statistics:\n" + "="*20 + f"\n"
        pred_text += f"Max value: {np.max(lbp_features):.4f}\n"
        pred_text += f"Min value: {np.min(lbp_features):.4f}\n"
        pred_text += f"Mean value: {np.mean(lbp_features):.4f}\n"
        pred_text += f"Std deviation: {np.std(lbp_features):.4f}\n"
        pred_text += f"Non-zero bins: {np.count_nonzero(lbp_features)}\n"
        
        # Top 3 features
        top_3_indices = np.argsort(lbp_features)[-3:]
        pred_text += f"\nTop 3 Features:\n" + "="*15 + f"\n"
        for i, idx in enumerate(reversed(top_3_indices)):
            pred_text += f"{i+1}. Bin {idx}: {lbp_features[idx]:.4f}\n"
        
        ax4.text(0.05, 0.95, pred_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save histogram
        histogram_path = os.path.join(self.evaluation_dir, f"{image_name}_lbp_analysis.png")
        plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Histogram saved: {histogram_path}")

def predict_fabric_with_steps(image_path, extractor, model, scaler, le):
    """Predict fabric class using LBP with step-by-step preprocessing"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return "Error: Cannot read image", None, None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create filename prefix from image name
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        filename_prefix = f"{image_name}"
        
        # Use the step-by-step preprocessor
        preprocessor = StepByStepPreprocessor()
        processed = preprocessor.preprocess_for_lbp_with_steps(image, filename_prefix)

        # Extract LBP features
        features, lbp_image = extractor.extract_lbp_features(processed)
        features_scaled = scaler.transform([features])
        pred_encoded = model.predict(features_scaled)
        pred_label = le.inverse_transform(pred_encoded)[0]

        return pred_label, features, lbp_image
        
    except Exception as e:
        return f"Error: {e}", None, None

def predict_fabric(image_path, extractor, model, scaler, le):
    """Original predict function (without step saving) - uses original preprocessing"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return "Error: Cannot read image"

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Use original preprocessing method directly
        processed = FabricLBPPreprocessor().preprocess_for_lbp(image)

        features, _ = extractor.extract_lbp_features(processed)
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
        
        # Initialize evaluator
        self.evaluator = LBPEvaluator(evaluation_dir)
        
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
        """Predict using all three models with step-by-step preprocessing and LBP evaluation"""
        models = [
            ("Random Forest", self.rf_model),
            ("SVM", self.svm_model),
            ("K-NN", self.knn_model)
        ]
        
        predictions = {}
        display_predictions = []
        
        # Show message that preprocessing is starting
        self.result_label.config(text="Starting step-by-step preprocessing...\nClose each window to continue to next step.")
        self.root.update()
        
        # Use step-by-step preprocessing for the first model
        first_pred, lbp_features, lbp_image = predict_fabric_with_steps(
            image_path, self.extractor, models[0][1], self.scaler, self.le
        )
        
        predictions[models[0][0]] = first_pred
        display_predictions.append(f"{models[0][0]}: {first_pred}")
        
        # Update GUI to show first prediction
        self.result_label.config(text=f"Step-by-step completed!\n{models[0][0]}: {first_pred}\n\nContinuing with other models...")
        self.root.update()
        
        # For the remaining models, use regular preprocessing
        for name, model in models[1:]:
            pred = predict_fabric(image_path, self.extractor, model, self.scaler, self.le)
            predictions[name] = pred
            display_predictions.append(f"{name}: {pred}")
        
        # Check consensus
        pred_values = [p for p in predictions.values() if not p.startswith("Error")]
        if len(set(pred_values)) == 1 and pred_values:
            display_predictions.append(f"\n✓ Consensus: {pred_values[0]}")
        
        # Save LBP evaluation if we have valid features
        if lbp_features is not None and lbp_image is not None:
            try:
                # Load original image for evaluation
                original_image = cv2.imread(image_path)
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                
                # Get image name
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                
                # Save evaluation
                self.evaluator.save_lbp_evaluation(
                    image_name, lbp_features, lbp_image, original_image, predictions
                )
                
            except Exception as e:
                display_predictions.append(f"\n⚠️ Evaluation save failed: {str(e)}")
        
        # Add file information
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        display_predictions.append(f"\nSteps saved: {image_name}_01 to {image_name}_06")
        display_predictions.append(f"Evaluation saved in: 5output_step_by_step/evaluations/")
        
        self.result_label.config(text="\n".join(display_predictions))

def main():
    root = tk.Tk()
    app = SimpleFabricGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()