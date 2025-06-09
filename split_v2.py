import os
import shutil
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
# Konstanta
DATASET_DIR = os.path.join(current_dir, "data")  # Path ke dataset
TRAIN_DIR = "train"  # Direktori output untuk train
TEST_DIR = "test"  # Direktori output untuk test
TEST_SIZE = 0.2  # Proporsi data test (20%)
RANDOM_SEED = 42  # Seed untuk reproduktibilitas


def create_split_directories():
    """Membuat direktori train dan test jika belum ada"""
    for directory in [TRAIN_DIR, TEST_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Membuat direktori: {directory}")
        else:
            print(f"Direktori {directory} sudah ada")


def split_dataset():
    """Memisahkan dataset menjadi train dan test"""
    # Set seed untuk reproduktibilitas
    random.seed(RANDOM_SEED)

    # Periksa apakah dataset ada
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Direktori dataset {DATASET_DIR} tidak ditemukan!")
        return

    # Dapatkan daftar kelas (001, 002, ..., 010)
    classes = [
        d
        for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ]
    classes.sort()  # Urutkan untuk konsistensi
    print(f"Kelas yang ditemukan: {classes}")

    # Proses setiap kelas
    for class_name in classes:
        class_dir = os.path.join(DATASET_DIR, class_name)
        # Dapatkan semua file gambar (.png)
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(".png")]
        total_images = len(image_files)
        if total_images == 0:
            print(f"Warning: Tidak ada gambar di kelas {class_name}")
            continue

        print(f"Memproses kelas {class_name}: {total_images} gambar")

        # Hitung jumlah gambar untuk test
        test_count = int(total_images * TEST_SIZE)
        train_count = total_images - test_count
        print(f"  Train: {train_count} gambar, Test: {test_count} gambar")

        # Acak daftar gambar
        random.shuffle(image_files)

        # Bagi menjadi train dan test
        test_files = image_files[:test_count]
        train_files = image_files[test_count:]

        # Buat subdirektori untuk kelas di train dan test
        train_class_dir = os.path.join(TRAIN_DIR, class_name)
        test_class_dir = os.path.join(TEST_DIR, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # Pindahkan file ke direktori train
        for file_name in train_files:
            src_path = os.path.join(class_dir, file_name)
            dst_path = os.path.join(train_class_dir, file_name)
            shutil.copy(src_path, dst_path)  # Gunakan copy untuk menjaga file asli

        # Pindahkan file ke direktori test
        for file_name in test_files:
            src_path = os.path.join(class_dir, file_name)
            dst_path = os.path.join(test_class_dir, file_name)
            shutil.copy(src_path, dst_path)

        print(f"  Selesai memisahkan kelas {class_name}")


def verify_split():
    """Verifikasi jumlah file di train dan test"""
    print("\nVerifikasi pembagian dataset:")
    total_train = 0
    total_test = 0
    for class_name in os.listdir(TRAIN_DIR):
        train_class_dir = os.path.join(TRAIN_DIR, class_name)
        test_class_dir = os.path.join(TEST_DIR, class_name)
        if os.path.isdir(train_class_dir) and os.path.isdir(test_class_dir):
            train_files = [
                f for f in os.listdir(train_class_dir) if f.lower().endswith(".png")
            ]
            test_files = [
                f for f in os.listdir(test_class_dir) if f.lower().endswith(".png")
            ]
            total_train += len(train_files)
            total_test += len(test_files)
            print(
                f"Kelas {class_name}: Train={len(train_files)}, Test={len(test_files)}"
            )
    print(f"Total: Train={total_train}, Test={total_test}")


def main():
    print("Memulai pembagian dataset...")
    create_split_directories()
    split_dataset()
    verify_split()
    print("\nPembagian dataset selesai!")
    print(f"Data train disimpan di: {TRAIN_DIR}")
    print(f"Data test disimpan di: {TEST_DIR}")
    print(
        "Pastikan untuk memperbarui path di kode.py untuk menggunakan direktori train!"
    )


if __name__ == "__main__":
    main()
