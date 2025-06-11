import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import clone
import cv2
from skimage import feature
import seaborn as sns
import joblib
from collections import Counter
from tqdm import tqdm

# Configuration
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, "3data_split_lbp")
TRAIN_DIR = os.path.join(dataset_dir, "train")
TEST_DIR = os.path.join(dataset_dir, "test")
OUTPUT_DIR = os.path.join(base_dir, "4model")

# Training parameters
RANDOM_STATE = 42
MAX_EPOCHS = 50
PATIENCE = 5

class LBPFeatureExtractor:
    def __init__(self, radius=3, n_points=24, method='uniform'):
        self.radius = radius
        self.n_points = n_points
        self.method = method
        
    def extract_lbp_features(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8)
        
        lbp = feature.local_binary_pattern(gray, self.n_points, self.radius, method=self.method)
        hist, _ = np.histogram(lbp.ravel(), bins=self.n_points + 2,
                             range=(0, self.n_points + 2), density=True)
        return hist

def load_and_process_data(data_dir, feature_extractor):
    print(f"Loading data from: {data_dir}")
    X = []
    y = []
    
    class_names = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    class_names.sort()

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        image_files = [f for f in os.listdir(class_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        for img_file in tqdm(image_files, desc=f"{class_name}", leave=False):
            try:
                img_path = os.path.join(class_dir, img_file)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                
                features = feature_extractor.extract_lbp_features(image)
                X.append(features)
                y.append(class_name)
                
            except Exception:
                continue

    print(f"Processed {len(X)} images from {len(class_names)} classes")
    return np.array(X), np.array(y), class_names

def train_with_early_stopping(model, X_train, y_train, model_name):
    best_score = 0
    patience_counter = 0
    best_model = None
    
    for epoch in range(1, MAX_EPOCHS + 1):
        # Set random state for this epoch
        if hasattr(model, 'random_state'):
            model.set_params(random_state=RANDOM_STATE + epoch)
        
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        
        if train_score > best_score:
            best_score = train_score
            # Create a clone of the fitted model
            best_model = clone(model)
            best_model.fit(X_train, y_train)  # Fit the clone
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break
    
    print(f"{model_name}: Best train accuracy {best_score:.3f} (epoch {epoch-patience_counter})")
    return best_model

def train_models(X, y, class_names):
    print("Training models with early stopping...")
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1),
        'SVM': SVC(kernel='rbf', random_state=RANDOM_STATE),
        'K-NN': KNeighborsClassifier(n_neighbors=5)
    }
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trained_models = {}
    
    for model_name, model in models.items():
        if model_name == 'Random Forest':
            best_model = train_with_early_stopping(model, X_scaled, y_encoded, model_name)
        else:
            model.fit(X_scaled, y_encoded)
            train_score = model.score(X_scaled, y_encoded)
            print(f"{model_name}: Train accuracy {train_score:.3f}")
            best_model = model
        
        trained_models[model_name] = best_model
        
        # Save model
        model_filename = f"{model_name.replace(' ', '_').replace('-', '')}_lbp_model.pkl"
        if model_name == "Random Forest":
            model_filename = "RF_lbp_model.pkl"
        elif model_name == "SVM":
            model_filename = "SVM_lbp_model.pkl" 
        elif model_name == "K-NN":
            model_filename = "KNN_lbp_model.pkl"
            
        joblib.dump(best_model, os.path.join(OUTPUT_DIR, model_filename))
    
    # Save scaler and label encoder
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "lbp_scaler.pkl"))
    joblib.dump(le, os.path.join(OUTPUT_DIR, "lbp_label_encoder.pkl"))
    
    return trained_models, scaler, le

def evaluate_models(test_dir, feature_extractor, models, scaler, le, class_names):
    print("Evaluating on test set...")
    
    X_test, y_test, _ = load_and_process_data(test_dir, feature_extractor)
    if len(X_test) == 0:
        return {}
    
    X_test_scaled = scaler.transform(X_test)
    y_test_encoded = le.transform(y_test)
    
    test_results = {}
    
    for model_name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test_encoded, y_pred)
        test_results[model_name] = {'accuracy': accuracy}
        
        print(f"{model_name}: Test accuracy {accuracy:.3f}")
        
        # Save confusion matrix
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test_encoded, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{model_name} - Accuracy: {accuracy:.3f}')
        plt.tight_layout()
        
        cm_filename = f"{model_name.replace(' ', '_').replace('-', '')}_confusion_matrix.png"
        plt.savefig(os.path.join(OUTPUT_DIR, cm_filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    return test_results

def create_performance_chart(test_results, output_dir):
    if not test_results:
        return
        
    fig, ax = plt.subplots(figsize=(8, 5))
    
    models = list(test_results.keys())
    accuracies = [test_results[model]['accuracy'] for model in models]
    
    bars = ax.bar(models, accuracies, alpha=0.8, color=['skyblue', 'lightcoral', 'lightgreen'])
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
               f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Model Performance Comparison')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    best_model = max(test_results.keys(), key=lambda x: test_results[x]['accuracy'])
    best_acc = test_results[best_model]['accuracy']
    ax.text(0.02, 0.98, f'Best: {best_model} ({best_acc:.3f})', 
           transform=ax.transAxes, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_performance.png"), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
        print("Error: Train or test directory not found!")
        return
    
    try:
        lbp_extractor = LBPFeatureExtractor(radius=3, n_points=24, method='uniform')
        
        X_train, y_train, class_names = load_and_process_data(TRAIN_DIR, lbp_extractor)
        if len(X_train) == 0:
            print("No training data processed!")
            return
        
        trained_models, scaler, le = train_models(X_train, y_train, class_names)
        test_results = evaluate_models(TEST_DIR, lbp_extractor, trained_models, scaler, le, class_names)
        create_performance_chart(test_results, OUTPUT_DIR)
        
        print(f"\nTraining completed! Models saved in: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()