import os
import pandas as pd
import numpy as np
import scipy.signal as signal
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

try:
    import pywt
except ImportError:
    pywt = None

class LightCurvePreprocessor:
    """
    Utility class for light curve preprocessing tasks including NaN interpolation,
    outlier clipping, normalization, stellar variability detrending, and noise-adaptive denoising.
    """
    
    @staticmethod
    def handle_missing_values(flux, method='interpolate'):
        """
        Handles missing values (NaNs) in the light curve.
        Methods:
        - 'interpolate': linear interpolation (default)
        - 'median': fill with median
        - 'zero': fill with zeros
        """
        flux = np.array(flux, dtype=float)
        nans = np.isnan(flux)
        if not np.any(nans):
            return flux
        
        if method == 'interpolate':
            x = np.arange(len(flux))
            flux[nans] = np.interp(x[nans], x[~nans], flux[~nans])
        elif method == 'median':
            median_val = np.nanmedian(flux)
            flux[nans] = median_val
        elif method == 'zero':
            flux[nans] = 0.0
        return flux

    @staticmethod
    def normalize_flux(flux, method='zscore'):
        """
        Normalizes the flux values of the light curve.
        Methods:
        - 'zscore': subtract mean and divide by std (default)
        - 'median': divide by median and subtract 1.0 (relative flux centered at 0)
        - 'minmax': scale between 0 and 1
        """
        if method == 'zscore':
            std_val = np.std(flux)
            if std_val == 0:
                return flux
            return (flux - np.mean(flux)) / std_val
        elif method == 'median':
            median_val = np.median(flux)
            if median_val == 0:
                return flux
            return flux / median_val - 1.0
        elif method == 'minmax':
            min_val = np.min(flux)
            max_val = np.max(flux)
            if max_val - min_val == 0:
                return flux
            return (flux - min_val) / (max_val - min_val)
        return flux

    @staticmethod
    def remove_outliers_sigma_clipping(flux, sigma=3.0, iters=2):
        """
        Applies iterative sigma clipping to remove extreme outliers and cosmic ray spikes.
        """
        clipped_flux = flux.copy()
        for _ in range(iters):
            mean = np.mean(clipped_flux)
            std = np.std(clipped_flux)
            if std == 0:
                break
            bad_idx = np.abs(clipped_flux - mean) > sigma * std
            clipped_flux[bad_idx] = mean
        return clipped_flux

    @staticmethod
    def remove_stellar_variability(flux, window_size=101):
        """
        Removes long-term stellar variability by subtracting a heavily median-filtered curve.
        """
        if window_size % 2 == 0:
            window_size += 1
        trend = signal.medfilt(flux, window_size)
        return flux - trend

    @staticmethod
    def estimate_noise(flux):
        """
        Estimates the noise level using standard deviation of successive differences.
        """
        diff = np.diff(flux)
        return float(np.std(diff) / np.sqrt(2))

    @staticmethod
    def savitzky_golay_filter(flux, window_size=15, polyorder=2):
        """
        Applies Savitzky-Golay filter to smooth high-frequency noise.
        """
        if window_size % 2 == 0:
            window_size += 1
        if len(flux) <= window_size:
            return flux
        return signal.savgol_filter(flux, window_size, polyorder)

    @staticmethod
    def moving_average(flux, window_size=5):
        """
        Applies moving average filter (acting as fallback for DWT denoising).
        """
        if window_size <= 1:
            return flux
        window = np.ones(int(window_size)) / float(window_size)
        return np.convolve(flux, window, 'same')

    @staticmethod
    def wavelet_denoising(flux, wavelet='db4', level=2):
        """
        Performs discrete wavelet transform (DWT) soft thresholding denoising.
        Falls back to moving average if PyWavelets (pywt) is missing.
        """
        if pywt is None:
            return LightCurvePreprocessor.moving_average(flux, window_size=5)
        
        coeffs = pywt.wavedec(flux, wavelet, mode='per')
        # Soft thresholding based on universal noise estimate
        sigma = (1/0.6745) * np.median(np.abs(coeffs[-1] - np.median(coeffs[-1])))
        threshold = sigma * np.sqrt(2 * np.log(len(flux)))
        
        new_coeffs = [coeffs[0]]
        for i in range(1, len(coeffs)):
            new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
            
        return pywt.waverec(new_coeffs, wavelet, mode='per')[:len(flux)]

    @staticmethod
    def hybrid_denoising(flux, wavelet='db4', level=2, median_kernel=5, sigma=3.0, iters=2):
        """
        Combines wavelet denoising, median filtering, and outlier clipping for highly noisy curves.
        """
        flux_denoised = LightCurvePreprocessor.wavelet_denoising(flux, wavelet=wavelet, level=level)
        
        if median_kernel % 2 == 0:
            median_kernel += 1
        if len(flux_denoised) > median_kernel:
            flux_med = signal.medfilt(flux_denoised, median_kernel)
        else:
            flux_med = flux_denoised
            
        flux_clipped = LightCurvePreprocessor.remove_outliers_sigma_clipping(flux_med, sigma=sigma, iters=iters)
        return flux_clipped

    @classmethod
    def preprocess(cls, flux, norm_method='zscore'):
        """
        Sequentially runs standard preprocessing:
        1. Gaps / NaNs interpolation
        2. Cosmic outlier rejection via sigma clipping
        3. Normalization (defaults to zscore)
        4. Slow stellar variability flattening (median filter subtraction)
        5. Adaptive high-frequency noise smoothing (Savitzky-Golay, Wavelet, or Hybrid based on noise estimate)
        """
        # 1. Gaps / NaNs
        flux = cls.handle_missing_values(flux, method='interpolate')
        
        # 2. Outlier rejection
        flux = cls.remove_outliers_sigma_clipping(flux, sigma=3.0, iters=2)
        
        # 3. Normalization
        flux = cls.normalize_flux(flux, method=norm_method)
        
        # 4. Stellar Variability removal
        flux = cls.remove_stellar_variability(flux, window_size=101)
        
        # 5. Denoising selector based on noise level
        noise = cls.estimate_noise(flux)
        if noise < 0.005:
            # Low noise: smooth using Savitzky-Golay
            flux = cls.savitzky_golay_filter(flux, window_size=15, polyorder=2)
        elif noise < 0.015:
            # Medium noise: smooth using Wavelet Transform
            flux = cls.wavelet_denoising(flux)
        else:
            # High noise: apply robust hybrid smoothing
            flux = cls.hybrid_denoising(flux)
            
        return flux

    @staticmethod
    def align_sequence_length(flux, target_len=2000):
        """
        Standardizes sequence dimensions by symmetrically cropping or edge-padding.
        """
        curr_len = len(flux)
        if curr_len > target_len:
            # Crop symmetrically from center
            start = (curr_len - target_len) // 2
            return flux[start:start + target_len]
        elif curr_len < target_len:
            # Pad with edge values
            pad_width = target_len - curr_len
            return np.pad(flux, (0, pad_width), mode='edge')
        return flux


class KeplerDataset:
    """
    Manages loading, label mapping, preprocessing, and stratified splitting of Kepler light curves.
    """
    def __init__(self, index_path="dataset_index.csv", target_len=2000, norm_method='zscore'):
        self.index_path = index_path
        self.target_len = target_len
        self.norm_method = norm_method

    def load_data(self):
        """
        Loads light curves listed in the index file, preprocesses them, and extracts labels.
        Labels:
        - 1.0 (transit / exoplanet candidate/confirmed)
        - 0.0 (non-transit false positive classes, eclipses, offsets, etc.)
        """
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index file {self.index_path} not found. Please run light_curve.py first.")
            
        df = pd.read_csv(self.index_path)
        x_data = []
        y_data = []
        kepids = []
        
        print(f"Loading and preprocessing light curves from {self.index_path}...")
        for idx, row in df.iterrows():
            filepath = row["file"]
            # Correct any backslash/slash mismatch for OS compatibility
            filepath = filepath.replace("\\", "/")
            
            if not os.path.exists(filepath):
                print(f"Warning: File {filepath} not found. Skipping.")
                continue
                
            data = np.load(filepath)
            if "flux" not in data:
                print(f"Warning: File {filepath} contains no 'flux' key. Skipping.")
                continue
            flux = data["flux"]
            
            # Preprocess and standardize length
            processed_flux = LightCurvePreprocessor.preprocess(flux, norm_method=self.norm_method)
            aligned_flux = LightCurvePreprocessor.align_sequence_length(processed_flux, target_len=self.target_len)
            
            # Map exoplanet transits (candidate or confirmed) to 1.0, and astronomical false positives to 0.0
            label = 1.0 if str(row["label"]).strip().lower() in ("transit", "candidate") else 0.0
            
            x_data.append(aligned_flux)
            y_data.append(label)
            kepids.append(row["kepid"])
            
        print(f"Successfully loaded {len(x_data)} valid light curve samples.")
        return np.array(x_data, dtype=np.float32), np.array(y_data, dtype=np.float32), np.array(kepids)

    def prepare_splits(self, test_size=0.15, val_size=0.15, seed=42):
        """
        Generates stratified Train, Validation, and Test splits.
        Formats X to shape (batch_size, target_len, 1) for deep learning architectures.
        """
        X, y, _ = self.load_data()
        
        if len(X) == 0:
            raise ValueError("Loaded 0 valid light curve samples. Confirm dataset folder contains valid .npz files.")
            
        # Stratified split: Train_Val and Test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        
        # Calculate validation split relative to the training split
        rel_val_size = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=rel_val_size, random_state=seed, stratify=y_train_val
        )
        
        # Add channel dimension (Batch, Length, 1)
        X_train = np.expand_dims(X_train, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class ModelEvaluator:
    """
    Computes classification reports, confusion matrices, and prints descriptive metrics.
    """
    
    @staticmethod
    def evaluate_predictions(y_true, y_pred, threshold=0.5):
        """
        Evaluates binary prediction values (probabilities or raw outputs) against ground truths.
        """
        y_pred_bin = np.where(y_pred >= threshold, 1.0, 0.0)
        
        acc = accuracy_score(y_true, y_pred_bin)
        prec = precision_score(y_true, y_pred_bin, zero_division=0)
        rec = recall_score(y_true, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true, y_pred_bin, zero_division=0)
        cm = confusion_matrix(y_true, y_pred_bin)
        
        metrics = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist()
        }
        
        print("\n" + "=" * 50)
        print("CLASSIFICATION PERFORMANCE METRICS")
        print("=" * 50)
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_bin, target_names=["non-transit", "transit"], zero_division=0))
        print("=" * 50)
        
        return metrics


if __name__ == "__main__":
    print("Testing Pipeline Utilities...")
    print(f"PyWavelets installed: {pywt is not None}")
    
    # Initialize dataset
    dataset_index = "dataset_index.csv"
    if not os.path.exists(dataset_index):
        # Fallback to check under modified datasets
        if os.path.exists("modified datasets/dataset_index.csv"):
            dataset_index = "modified datasets/dataset_index.csv"
        elif os.path.exists("dataset_index.csv"):
            dataset_index = "dataset_index.csv"
            
    print(f"Using dataset index: {dataset_index}")
    try:
        dataset = KeplerDataset(index_path=dataset_index, target_len=2000)
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = dataset.prepare_splits(test_size=0.15, val_size=0.15)
        
        print("\nDataset split successfully!")
        print(f"Train shapes:      X={x_train.shape}, Y={y_train.shape}")
        print(f"Validation shapes: X={x_val.shape}, Y={y_val.shape}")
        print(f"Test shapes:        X={x_test.shape}, Y={y_test.shape}")
        
        print(f"Positive transits counts - Train: {np.sum(y_train)}, Val: {np.sum(y_val)}, Test: {np.sum(y_test)}")
        
        # Test evaluator on dummy predictions
        print("\nRunning ModelEvaluator dry-run...")
        dummy_preds = np.random.uniform(0, 1, size=len(y_test))
        ModelEvaluator.evaluate_predictions(y_test, dummy_preds)
        
    except Exception as e:
        print(f"\nSelf-test error: {e}")
