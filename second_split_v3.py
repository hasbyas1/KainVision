import os
import shutil
import random
from tqdm import tqdm

# Current working directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Input directory
INPUT_DIR = os.path.join(base_dir, "2preprocessed_output_lbp") # Path to preprocessed dataset

# Output directory
dataset_dir = os.path.join(base_dir, "3data_split_lbp")  # Path to output dataset folder
TRAIN_DIR = os.path.join(dataset_dir, "train") # Path to output train dataset
TEST_DIR = os.path.join(dataset_dir, "test") # Path to output test dataset

print("Path to input dataset:", INPUT_DIR)
print("Path to train dataset:", TRAIN_DIR)
print("Path to test dataset:", TEST_DIR)

TEST_SIZE = 0.2  # 20% for test
RANDOM_SEED = 42  # For reproducibility

# ====================================================

class PreprocessedDatasetSplitter:
    """
    Split preprocessed dataset into train and test
    """
    
    def __init__(self, input_dir=INPUT_DIR, 
                 train_dir=TRAIN_DIR, test_dir=TEST_DIR, test_size=TEST_SIZE, random_seed=RANDOM_SEED):
        self.input_dir = INPUT_DIR
        self.train_dir = TRAIN_DIR
        self.test_dir = TEST_DIR
        self.test_size = TEST_SIZE
        self.random_seed = RANDOM_SEED
        self.classes = self._get_classes()
        
    def _get_classes(self):
        """Get classes from preprocessed directory"""
        if not os.path.exists(self.input_dir):
            raise ValueError(f"Preprocessed directory {self.input_dir} not found!")
        
        classes = [d for d in os.listdir(self.input_dir) 
                  if os.path.isdir(os.path.join(self.input_dir, d))]
        classes.sort()
        return classes
    
    def get_image_files(self, class_dir):
        """Get all image files from class directory"""
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        image_files = []
        for ext in extensions:
            image_files.extend([f for f in os.listdir(class_dir) 
                              if f.lower().endswith(ext.lower())])
        return image_files
    
    def create_directories(self):
        """Create train and test directories"""
        # Remove existing directories
        for directory in [self.train_dir, self.test_dir]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
        
        # Create new directories for each class
        for directory in [self.train_dir, self.test_dir]:
            for class_name in self.classes:
                os.makedirs(os.path.join(directory, class_name), exist_ok=True)
    
    def split_dataset(self):
        """Split dataset into train and test"""
        print(f"Splitting dataset from {self.input_dir}")
        print(f"Classes: {self.classes}")
        print(f"Train/Test split: {int((1-self.test_size)*100)}% / {int(self.test_size*100)}%")
        
        # Set random seed
        random.seed(self.random_seed)
        
        # Create directories
        self.create_directories()
        
        total_train = 0
        total_test = 0
        
        # Process each class
        for class_name in self.classes:
            class_dir = os.path.join(self.input_dir, class_name)
            image_files = self.get_image_files(class_dir)
            
            if not image_files:
                print(f"Warning: No images in {class_name}")
                continue
            
            # Calculate split
            total_images = len(image_files)
            test_count = int(total_images * self.test_size)
            train_count = total_images - test_count
            
            # Shuffle files
            random.shuffle(image_files)
            
            # Split files
            test_files = image_files[:test_count]
            train_files = image_files[test_count:]
            
            # Copy files to train directory
            train_class_dir = os.path.join(self.train_dir, class_name)
            for file_name in tqdm(train_files, desc=f"{class_name} train"):
                src_path = os.path.join(class_dir, file_name)
                dst_path = os.path.join(train_class_dir, file_name)
                shutil.copy2(src_path, dst_path)
            
            # Copy files to test directory
            test_class_dir = os.path.join(self.test_dir, class_name)
            for file_name in tqdm(test_files, desc=f"{class_name} test"):
                src_path = os.path.join(class_dir, file_name)
                dst_path = os.path.join(test_class_dir, file_name)
                shutil.copy2(src_path, dst_path)
            
            total_train += train_count
            total_test += test_count
            
            print(f"{class_name}: {train_count} train, {test_count} test")
        
        print(f"\nSplit completed!")
        print(f"Total train: {total_train} images -> {self.train_dir}")
        print(f"Total test: {total_test} images -> {self.test_dir}")
    
    def verify_split(self):
        """Verify the split results"""
        print(f"\nVerifying split...")
        
        for directory, dir_name in [(self.train_dir, "train"), (self.test_dir, "test")]:
            if not os.path.exists(directory):
                print(f"Warning: {directory} not found!")
                continue
            
            total_images = 0
            for class_name in self.classes:
                class_dir = os.path.join(directory, class_name)
                if os.path.exists(class_dir):
                    image_files = self.get_image_files(class_dir)
                    total_images += len(image_files)
                    print(f"{dir_name}/{class_name}: {len(image_files)} images")
            
            print(f"Total {dir_name}: {total_images} images")
        
        print("Verification completed!")


def main():
    """
    Split preprocessed dataset
    """
    # Initialize splitter
    splitter = PreprocessedDatasetSplitter()
    
    # Split dataset
    splitter.split_dataset()
    
    # Verify results
    splitter.verify_split()


if __name__ == "__main__":
    main()