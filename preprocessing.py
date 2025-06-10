import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm

class FabricLBPPreprocessor:
    """
    Simple LBP preprocessing pipeline for fabric classification
    """
    
    def __init__(self, input_dir="data", output_dir="preprocessed_data_lbp"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.classes = self._get_classes()
        
    def _get_classes(self):
        """Get fabric classes from input directory"""
        classes = [d for d in os.listdir(self.input_dir) 
                  if os.path.isdir(os.path.join(self.input_dir, d))]
        classes.sort()
        return classes
    
    def resize_image(self, image, scale_factor=0.9):
        """Resize image 10% smaller then to 224x224"""
        height, width = image.shape[:2]
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        resized = cv2.resize(image, (new_width, new_height))
        return cv2.resize(resized, (224, 224))
    
    def adaptive_normalization(self, image, window_size=64):
        """Adaptive normalization for lighting consistency"""
        if window_size % 2 == 0:
            window_size += 1
        
        kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
        local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
        local_sq_mean = cv2.filter2D((image.astype(np.float32))**2, -1, kernel)
        local_var = local_sq_mean - local_mean**2
        local_std = np.sqrt(np.maximum(local_var, 1e-10))
        
        normalized = (image.astype(np.float32) - local_mean) / local_std * 50 + 127
        normalized = np.clip(normalized, 0, 255)
        return normalized.astype(np.uint8)
    
    def preprocess_for_lbp(self, image):
        """
        LBP preprocessing pipeline:
        Resize → Grayscale → Light Blur → Normalization → Mild Contrast
        """
        # Resize
        processed = self.resize_image(image, scale_factor=0.9)
        
        # Convert to grayscale
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        
        # Light Gaussian blur
        processed = cv2.GaussianBlur(processed, (3, 3), 0.8)
        
        # Adaptive normalization
        processed = self.adaptive_normalization(processed)
        
        # Mild CLAHE contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        processed = clahe.apply(processed)
        
        return processed
    
    def process_dataset(self):
        """Process entire dataset"""
        print(f"Processing {len(self.classes)} classes from {self.input_dir}")
        print(f"Classes: {self.classes}")
        
        # Remove and recreate output directory
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        
        # Create directories
        for class_name in self.classes:
            os.makedirs(os.path.join(self.output_dir, class_name), exist_ok=True)
        
        # Process images
        total_processed = 0
        for class_name in self.classes:
            input_class_dir = os.path.join(self.input_dir, class_name)
            output_class_dir = os.path.join(self.output_dir, class_name)
            
            # Get image files
            extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
            image_files = []
            for ext in extensions:
                image_files.extend([f for f in os.listdir(input_class_dir) 
                                  if f.lower().endswith(ext.lower())])
            
            # Process each image
            for img_file in tqdm(image_files, desc=f"{class_name}"):
                try:
                    img_path = os.path.join(input_class_dir, img_file)
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                    
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    processed = self.preprocess_for_lbp(image)
                    
                    output_path = os.path.join(output_class_dir, img_file)
                    cv2.imwrite(output_path, processed)
                    total_processed += 1
                    
                except Exception as e:
                    continue
        
        print(f"Processed {total_processed} images -> {self.output_dir}")


def main():
    # Simple execution
    preprocessor = FabricLBPPreprocessor(input_dir="1base_dataset", output_dir="2preprocessed_output_lbp")
    preprocessor.process_dataset()


if __name__ == "__main__":
    main()