import os
import numpy as np
import pandas as pd
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
EVALUATION_DIR = os.path.join(OUTPUT_DIR, "evaluations")

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

def save_features_to_excel(features_data, filename):
    """Simpan features ke Excel dengan format yang rapi"""
    try:
        os.makedirs(EVALUATION_DIR, exist_ok=True)
        filepath = os.path.join(EVALUATION_DIR, filename)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            features_data.to_excel(writer, sheet_name='LBP_Features', index=False)
            
            summary = features_data.groupby('class').agg({
                'image_name': 'count',
                'feature_000': 'mean',
                'feature_001': 'mean'
            }).round(4)
            summary.columns = ['Total_Images', 'Avg_Feature_000', 'Avg_Feature_001']
            summary.to_excel(writer, sheet_name='Summary')
        
        print("Features berhasil disimpan:", filepath)
        return True
        
    except ImportError:
        print("openpyxl tidak tersedia. Install dengan: pip install openpyxl")
        csv_path = filepath.replace('.xlsx', '.csv')
        features_data.to_csv(csv_path, index=False)
        print("Fallback: Features disimpan sebagai CSV:", csv_path)
        return False
    except Exception as e:
        print("Error menyimpan Excel:", str(e))
        return False

def create_histogram_visualizations(features_data, data_type):
    """Buat visualisasi histogram LBP - hanya 2 grafik"""
    print("Creating histogram visualizations for", data_type, "data...")
    
    os.makedirs(EVALUATION_DIR, exist_ok=True)
    feature_cols = [col for col in features_data.columns if col.startswith('feature_')]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    title_text = 'LBP Histogram Analysis - ' + data_type.title() + ' Data'
    fig.suptitle(title_text, fontsize=16, fontweight='bold')
    
    # Plot 1: Average histogram per class
    ax1 = axes[0]
    classes = features_data['class'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    
    for i, class_name in enumerate(classes):
        class_data = features_data[features_data['class'] == class_name]
        avg_features = class_data[feature_cols].mean()
        
        label_text = class_name + ' (n=' + str(len(class_data)) + ')'
        ax1.plot(range(len(avg_features)), avg_features, 'o-', 
                label=label_text, color=colors[i], linewidth=2, markersize=4)
    
    ax1.set_title('Average LBP Histogram by Class', fontweight='bold')
    ax1.set_xlabel('LBP Bin Index')
    ax1.set_ylabel('Average Feature Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Sample individual histograms
    ax2 = axes[1]
    sample_size = min(5, len(features_data))
    sample_data = features_data.sample(n=sample_size, random_state=42)
    
    for i, (_, row) in enumerate(sample_data.iterrows()):
        features = row[feature_cols].values
        label_text = row['class'] + ' - ' + row['image_name'][:10] + '...'
        ax2.plot(range(len(features)), features, 'o-', 
                label=label_text, alpha=0.7, linewidth=2, markersize=3)
    
    title_text2 = 'Sample Individual Histograms (n=' + str(sample_size) + ')'
    ax2.set_title(title_text2, fontweight='bold')
    ax2.set_xlabel('Feature Index')
    ax2.set_ylabel('Feature Value')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    histogram_path = os.path.join(EVALUATION_DIR, data_type + '_lbp_histograms.png')
    plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Histogram visualizations saved:", histogram_path)
    create_detailed_class_histograms(features_data, data_type)

def create_detailed_class_histograms(features_data, data_type):
    """Buat histogram detail per kelas"""
    feature_cols = [col for col in features_data.columns if col.startswith('feature_')]
    classes = features_data['class'].unique()
    
    n_classes = len(classes)
    cols = min(3, n_classes)
    rows = (n_classes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_classes == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    title_text = 'LBP Histograms by Class - ' + data_type.title() + ' Data'
    fig.suptitle(title_text, fontsize=16, fontweight='bold')
    
    for i, class_name in enumerate(classes):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        class_data = features_data[features_data['class'] == class_name]
        mean_features = class_data[feature_cols].mean()
        std_features = class_data[feature_cols].std()
        
        x = range(len(feature_cols))
        ax.plot(x, mean_features, 'o-', linewidth=2, markersize=5, label='Mean')
        ax.fill_between(x, mean_features - std_features, mean_features + std_features, 
                       alpha=0.3, label='±1 STD')
        
        title_text2 = class_name + '\n(n=' + str(len(class_data)) + ' images)'
        ax.set_title(title_text2, fontweight='bold')
        ax.set_xlabel('LBP Bin Index')
        ax.set_ylabel('Feature Value')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        max_val = mean_features.max()
        min_val = mean_features.min()
        text_content = 'Max: ' + str(round(max_val, 3)) + '\nMin: ' + str(round(min_val, 3))
        ax.text(0.02, 0.98, text_content, transform=ax.transAxes, fontsize=8, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide empty subplots
    for i in range(n_classes, rows * cols):
        if rows > 1:
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)
        elif cols > 1 and i < len(axes):
            axes[i].set_visible(False)
    
    plt.tight_layout()
    class_histogram_path = os.path.join(EVALUATION_DIR, data_type + '_class_histograms.png')
    plt.savefig(class_histogram_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Class histograms saved:", class_histogram_path)

def print_histogram_statistics(features_data, data_type):
    """Print statistik histogram ke console"""
    print("\nHISTOGRAM STATISTICS -", data_type.upper(), "DATA")
    print("=" * 60)
    
    feature_cols = [col for col in features_data.columns if col.startswith('feature_')]
    
    print("Total images:", len(features_data))
    print("Feature dimensions:", len(feature_cols))
    print("Classes:", list(features_data['class'].unique()))
    
    print("\nPER-CLASS HISTOGRAM STATISTICS:")
    print("-" * 40)
    
    for class_name in features_data['class'].unique():
        class_data = features_data[features_data['class'] == class_name]
        class_features = class_data[feature_cols]
        
        print("\n", class_name, "(n=" + str(len(class_data)) + "):")
        
        mean_values = class_features.mean().head(5)
        mean_str = ', '.join([str(round(x, 3)) for x in mean_values])
        print("   Mean histogram: [" + mean_str + "...]")
        
        max_val = class_features.max().max()
        min_val = class_features.min().min()
        print("   Max feature value:", round(max_val, 4))
        print("   Min feature value:", round(min_val, 4))
        
        variance_series = class_features.var()
        most_variable_feature = variance_series.idxmax()
        bin_number = most_variable_feature.split('_')[1]
        max_variance = variance_series.max()
        print("   Most variable bin: feature_" + bin_number + " (var=" + str(round(max_variance, 6)) + ")")
        
        top_features = class_features.mean().nlargest(3)
        top_bins_list = []
        for feature_name, value in top_features.items():
            bin_num = feature_name.split('_')[1]
            top_bins_list.append(bin_num + "(" + str(round(value, 3)) + ")")
        
        print("   Top 3 bins:", ', '.join(top_bins_list))
    
    print("\nCROSS-CLASS COMPARISON:")
    print("-" * 30)
    
    class_means = features_data.groupby('class')[feature_cols].mean()
    overall_variance = class_means.var()
    most_discriminative = overall_variance.nlargest(5)
    
    print("Most discriminative features (highest variance between classes):")
    for feature_name, variance in most_discriminative.items():
        bin_idx = feature_name.split('_')[1]
        print("   Bin " + bin_idx + ": variance = " + str(round(variance, 6)))
    
    print("=" * 60)

def load_and_process_data(data_dir, feature_extractor, data_type="train"):
    print("Loading data from:", data_dir)
    X = []
    y = []
    features_data = []
    
    class_names = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    class_names.sort()

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        image_files = [f for f in os.listdir(class_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        for img_file in tqdm(image_files, desc="Processing " + class_name, leave=False):
            try:
                img_path = os.path.join(class_dir, img_file)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                
                features = feature_extractor.extract_lbp_features(image)
                X.append(features)
                y.append(class_name)
                
                row_data = {
                    'image_name': img_file,
                    'class': class_name,
                    'data_type': data_type
                }
                
                for i, feature_val in enumerate(features):
                    feature_key = 'feature_' + str(i).zfill(3)
                    row_data[feature_key] = round(feature_val, 6)
                
                features_data.append(row_data)
                
            except Exception:
                continue

    print("Processed", len(X), "images from", len(class_names), "classes")
    
    if len(features_data) > 0:
        df = pd.DataFrame(features_data)
        save_features_to_excel(df, data_type + "_lbp_features.xlsx")
        print_histogram_statistics(df, data_type)
        create_histogram_visualizations(df, data_type)
        
        print("\nSummary", data_type, "data:")
        for class_name, count in df['class'].value_counts().items():
            print("  ", class_name + ":", count, "images")
    
    return np.array(X), np.array(y), class_names

def train_with_early_stopping(model, X_train, y_train, model_name):
    best_score = 0
    patience_counter = 0
    best_model = None
    
    for epoch in range(1, MAX_EPOCHS + 1):
        if hasattr(model, 'random_state'):
            model.set_params(random_state=RANDOM_STATE + epoch)
        
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        
        if train_score > best_score:
            best_score = train_score
            best_model = clone(model)
            best_model.fit(X_train, y_train)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break
    
    print(model_name + ": Best train accuracy " + str(round(best_score, 3)) + " (epoch " + str(epoch-patience_counter) + ")")
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
        print("Training", model_name + "...")
        
        if model_name == 'Random Forest':
            best_model = train_with_early_stopping(model, X_scaled, y_encoded, model_name)
        else:
            model.fit(X_scaled, y_encoded)
            train_score = model.score(X_scaled, y_encoded)
            print(model_name + ": Train accuracy " + str(round(train_score, 3)))
            best_model = model
        
        trained_models[model_name] = best_model
        
        if model_name == "Random Forest":
            model_filename = "RF_lbp_model.pkl"
        elif model_name == "SVM":
            model_filename = "SVM_lbp_model.pkl" 
        elif model_name == "K-NN":
            model_filename = "KNN_lbp_model.pkl"
        else:
            model_filename = model_name.replace(' ', '_').replace('-', '') + "_lbp_model.pkl"
            
        joblib.dump(best_model, os.path.join(OUTPUT_DIR, model_filename))
        print(model_name, "saved:", model_filename)
    
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "lbp_scaler.pkl"))
    joblib.dump(le, os.path.join(OUTPUT_DIR, "lbp_label_encoder.pkl"))
    print("Scaler and label encoder saved")
    
    return trained_models, scaler, le

def evaluate_models(test_dir, feature_extractor, models, scaler, le, class_names):
    print("Evaluating on test set...")
    
    X_test, y_test, _ = load_and_process_data(test_dir, feature_extractor, "test")
    if len(X_test) == 0:
        return {}
    
    X_test_scaled = scaler.transform(X_test)
    y_test_encoded = le.transform(y_test)
    
    test_results = {}
    
    print("\n=== Model Performance Metrics ===")
    for model_name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        
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
        
        print(model_name + ": Acc=" + str(round(accuracy, 3)) + 
              " Prec=" + str(round(precision, 3)) + 
              " Rec=" + str(round(recall, 3)) + 
              " F1=" + str(round(f1, 3)))
        
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test_encoded, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        
        title_text = model_name + ' Confusion Matrix\nAccuracy: ' + str(round(accuracy, 3))
        plt.title(title_text, fontweight='bold')
        plt.xlabel('Predicted Label', fontweight='bold')
        plt.ylabel('True Label', fontweight='bold')
        plt.tight_layout()
        
        cm_filename = model_name.replace(' ', '_').replace('-', '') + "_confusion_matrix.png"
        plt.savefig(os.path.join(EVALUATION_DIR, cm_filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(model_name, "confusion matrix saved")
    
    print("=" * 45)
    return test_results

def create_compact_visualizations(test_results, output_dir):
    if not test_results:
        return
    
    os.makedirs(EVALUATION_DIR, exist_ok=True)
    
    models = list(test_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # Performance Summary Table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Rank']
    
    f1_scores = [(model, test_results[model]['f1']) for model in models]
    f1_scores.sort(key=lambda x: x[1], reverse=True)
    ranks = {model: i+1 for i, (model, _) in enumerate(f1_scores)}
    
    for model in models:
        row = [
            model,
            str(round(test_results[model]['accuracy'], 3)),
            str(round(test_results[model]['precision'], 3)),
            str(round(test_results[model]['recall'], 3)),
            str(round(test_results[model]['f1'], 3)),
            "#" + str(ranks[model])
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    best_model = max(models, key=lambda x: test_results[x]['f1'])
    best_row = models.index(best_model) + 1
    for i in range(len(headers)):
        table[(best_row, i)].set_facecolor('#FFD93D')
        table[(best_row, i)].set_text_props(weight='bold')
    
    plt.title('Model Performance Summary Table', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(EVALUATION_DIR, "performance_summary_table.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Grouped Metrics Comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [test_results[model][metric] for model in models]
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize(), 
                     color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   str(round(val, 3)), ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Models', fontweight='bold', fontsize=12)
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_title('Model Performance Comparison - All Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    best_model = max(models, key=lambda x: test_results[x]['f1'])
    best_f1 = test_results[best_model]['f1']
    text_content = 'Best Model: ' + best_model + '\nF1-Score: ' + str(round(best_f1, 3))
    ax.text(0.02, 0.98, text_content, transform=ax.transAxes, fontweight='bold', fontsize=11,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFD93D', alpha=0.8),
           verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(EVALUATION_DIR, "grouped_metrics_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Performance visualizations saved to:", EVALUATION_DIR)

def main():
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
        print("Error: Train or test directory not found!")
        return
    
    print("Starting LBP Feature Extraction and Model Training...")
    print("=" * 60)
    
    try:
        lbp_extractor = LBPFeatureExtractor(radius=3, n_points=24, method='uniform')
        
        print("\nProcessing Training Data...")
        X_train, y_train, class_names = load_and_process_data(TRAIN_DIR, lbp_extractor, "train")
        
        if len(X_train) == 0:
            print("No training data processed!")
            return
        
        print("\nTraining Models...")
        trained_models, scaler, le = train_models(X_train, y_train, class_names)
        
        print("\nEvaluating Models...")
        test_results = evaluate_models(TEST_DIR, lbp_extractor, trained_models, scaler, le, class_names)
        
        if test_results:
            create_compact_visualizations(test_results, OUTPUT_DIR)
        
        print("\nTRAINING COMPLETED!")
        print("=" * 60)
        print("Models disimpan di:", OUTPUT_DIR)
        print("Evaluasi disimpan di:", EVALUATION_DIR)
        print("\nFiles yang dihasilkan:")
        print("   Excel Features (di evaluation_images/):")
        print("   - train_lbp_features.xlsx")
        print("   - test_lbp_features.xlsx")
        print("\n   Performance Visualizations (di evaluation_images/):")
        print("   - performance_summary_table.png")
        print("   - grouped_metrics_comparison.png")
        print("   - confusion matrices for each model")
        print("\n   Models (di 4model/):")
        print("   - RF_lbp_model.pkl")
        print("   - SVM_lbp_model.pkl") 
        print("   - KNN_lbp_model.pkl")
        print("   - lbp_scaler.pkl")
        print("   - lbp_label_encoder.pkl")
        
    except Exception as e:
        print("Error:", str(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()