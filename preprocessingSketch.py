def adaptive_quantization(image, num_levels=8):
    """
    Adaptive quantization berdasarkan histogram
    Membagi berdasarkan distribusi pixel yang lebih merata
    """
    # Hitung histogram
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    
    # Hitung cumulative distribution
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]
    
    # Buat mapping berdasarkan CDF
    mapping = np.round(cdf_normalized * (num_levels - 1)).astype(np.uint8)
    
    # Apply mapping
    quantized = mapping[image]
    
    return quantized

def clahe_enhancement(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE)
    Lebih baik dari histogram equalization standard
    
    Parameters:
    - image: input grayscale image
    - clip_limit: batas kontras (1.0-4.0, default: 2.0)
    - tile_grid_size: ukuran grid untuk adaptasi lokal (default: 8x8)
    
    Returns:
    - CLAHE enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def adaptive_normalization(image, window_size=64):
    """
    Adaptive normalization berdasarkan neighborhood lokal
    Bagus untuk citra dengan pencahayaan sangat tidak merata
    """
    # Pastikan window_size ganjil
    if window_size % 2 == 0:
        window_size += 1
    
    # Hitung mean dan std lokal
    kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
    local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
    
    # Hitung variance lokal
    local_sq_mean = cv2.filter2D((image.astype(np.float32))**2, -1, kernel)
    local_var = local_sq_mean - local_mean**2
    local_std = np.sqrt(np.maximum(local_var, 1e-10))  # Hindari pembagian nol
    
    # Normalisasi adaptif
    normalized = (image.astype(np.float32) - local_mean) / local_std * 50 + 127
    normalized = np.clip(normalized, 0, 255)
    
    return normalized.astype(np.uint8)

def opencv_bgr_to_gray(image):
    """
    Konversi BGR ke Grayscale menggunakan OpenCV
    Formula: 0.299*R + 0.587*G + 0.114*B (weighted average)
    
    Parameters:
    - image: input BGR image (dari cv2.imread)
    
    Returns:
    - grayscale image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def adaptive_gaussian_blur(image, window_size=15, noise_threshold=20):
    """
    Adaptive Gaussian Blur
    Blur lebih kuat di area noise, minimal di area edge
    
    Parameters:
    - image: input image
    - window_size: ukuran window untuk analisis lokal
    - noise_threshold: threshold untuk deteksi noise
    """
    # Hitung variance lokal untuk deteksi noise
    kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
    local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
    local_sq_mean = cv2.filter2D((image.astype(np.float32))**2, -1, kernel)
    local_var = local_sq_mean - local_mean**2
    
    # Adaptive blurring
    result = image.copy().astype(np.float32)
    
    # Area dengan variance tinggi (noise) → blur lebih kuat
    high_noise_mask = local_var > noise_threshold
    result[high_noise_mask] = cv2.GaussianBlur(image, (5, 5), 1.5)[high_noise_mask]
    
    # Area dengan variance rendah (smooth) → blur ringan
    low_noise_mask = local_var <= noise_threshold
    result[low_noise_mask] = cv2.GaussianBlur(image, (3, 3), 0.5)[low_noise_mask]
    
    return result.astype(np.uint8)

## Resize belum ada