import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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
    
    print("\n=== Model Performance Metrics ===")
    for model_name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        
        # Calculate all metrics
        accuracy = accuracy_score(y_test_encoded, y_pred)
        precision = precision_score(y_test_encoded, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test_encoded, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test_encoded, y_pred, average='macro', zero_division=0)
        
        test_results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # Print compact metrics
        print(f"{model_name}: Acc={accuracy:.3f} Prec={precision:.3f} Rec={recall:.3f} F1={f1:.3f}")
        
        # Save confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test_encoded, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{model_name} Confusion Matrix\nAccuracy: {accuracy:.3f}', fontweight='bold')
        plt.xlabel('Predicted Label', fontweight='bold')
        plt.ylabel('True Label', fontweight='bold')
        plt.tight_layout()
        
        cm_filename = f"{model_name.replace(' ', '_').replace('-', '')}_confusion_matrix.png"
        plt.savefig(os.path.join(OUTPUT_DIR, cm_filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("=" * 45)
    
    return test_results

def create_compact_visualizations(test_results, output_dir):
    """Create the 3 requested visualizations: table, confusion matrices (already done), and grouped metrics"""
    if not test_results:
        return
    
    models = list(test_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # 1. Performance Summary Table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
    table_data = []
    headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Rank']
    
    # Calculate ranks based on F1-score
    f1_scores = [(model, test_results[model]['f1']) for model in models]
    f1_scores.sort(key=lambda x: x[1], reverse=True)
    ranks = {model: i+1 for i, (model, _) in enumerate(f1_scores)}
    
    for model in models:
        row = [
            model,
            f"{test_results[model]['accuracy']:.3f}",
            f"{test_results[model]['precision']:.3f}",
            f"{test_results[model]['recall']:.3f}",
            f"{test_results[model]['f1']:.3f}",
            f"#{ranks[model]}"
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best performance rows
    best_model = max(models, key=lambda x: test_results[x]['f1'])
    best_row = models.index(best_model) + 1
    for i in range(len(headers)):
        table[(best_row, i)].set_facecolor('#FFD93D')
        table[(best_row, i)].set_text_props(weight='bold')
    
    plt.title('Model Performance Summary Table', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(output_dir, "performance_summary_table.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Grouped Metrics Comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [test_results[model][metric] for model in models]
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize(), 
                     color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Models', fontweight='bold', fontsize=12)
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_title('Model Performance Comparison - All Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Add best model annotation
    best_model = max(models, key=lambda x: test_results[x]['f1'])
    best_f1 = test_results[best_model]['f1']
    ax.text(0.02, 0.98, f'Best Model: {best_model}\nF1-Score: {best_f1:.3f}', 
           transform=ax.transAxes, fontweight='bold', fontsize=11,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFD93D', alpha=0.8),
           verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grouped_metrics_comparison.png"), dpi=300, bbox_inches='tight')
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
        
        # Create compact visualizations
        create_compact_visualizations(test_results, OUTPUT_DIR)
        
        print(f"\nTraining completed! Models and visualizations saved in: {OUTPUT_DIR}")
        print("\nGenerated visualization files:")
        print("- performance_summary_table.png")
        print("- grouped_metrics_comparison.png")
        print("- Confusion matrices for each model:")
        for model_name in trained_models.keys():
            cm_filename = f"{model_name.replace(' ', '_').replace('-', '')}_confusion_matrix.png"
            print(f"  - {cm_filename}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()