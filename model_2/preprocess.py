import numpy as np

def remove_outliers(time, flux, sigma=5.0):
    # Removes outliers that are beyond 5 standard deviations from the mean flux.
    if len(flux) == 0:
        return time, flux
    
    mean_val = np.mean(flux)
    std_val = np.std(flux)
    
    if std_val == 0:
        return time, flux
        
    mask = np.abs(flux - mean_val) <= sigma * std_val
    return time[mask], flux[mask]


def phase_fold(time, period, epoch):
    # Folds the light curve time coordinates into orbital phase coordinates:
    # phi = (time - epoch) / period
    # phi = phi mod 1
    # phi = phi - 1 if phi > 0.5
    # This maps all phases into the range [-0.5, 0.5].
    # Map to [0, 1) range
    phase = ((time - epoch) / period) % 1.0
    # Map to [-0.5, 0.5] range
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    return phase


def bin_light_curve(phase, flux, num_bins, min_phase, max_phase):
    # Bins the phase-folded light curve into a fixed number of bins.
    # Uses median flux in each bin. Empty bins are linearly interpolated.
    # 1. Create bin edges
    edges = np.linspace(min_phase, max_phase, num_bins + 1)
    
    # 2. Digitize phase values into bin indices (0-indexed)
    bin_indices = np.digitize(phase, edges) - 1
    
    # 3. Calculate median flux per bin
    binned_flux = np.zeros(num_bins)
    for i in range(num_bins):
        mask = (bin_indices == i)
        if np.any(mask):
            binned_flux[i] = np.median(flux[mask])
        else:
            binned_flux[i] = np.nan
            
    # 4. Linearly interpolate empty bins (NaNs)
    nans = np.isnan(binned_flux)
    if np.all(nans):
        # Fallback: if all bins are empty, return a zero vector
        return np.zeros(num_bins)
    elif np.any(nans):
        x = np.arange(num_bins)
        binned_flux[nans] = np.interp(x[nans], x[~nans], binned_flux[~nans])
        
    return binned_flux


def normalize_view(flux_view):
    # Median-subtracts and standard-deviation normalizes a binned view.
    median_val = np.median(flux_view)
    std_val = np.std(flux_view)
    
    if std_val > 0:
        return (flux_view - median_val) / std_val
    else:
        return flux_view - median_val


def augment_view(flux_view):
    # Applies data augmentations during training.
    # Each of the following is applied with a probability of 0.5:
    # - Additive Gaussian noise (std = 0.008)
    # - Random phase shift (up to 30 bins)
    # - Multiplicative flux scaling (scale factor in range [0.98, 1.02])
    augmented = flux_view.copy()
    
    # 1. Additive Gaussian noise (sigma = 0.008)
    if np.random.rand() < 0.5:
        noise = np.random.normal(0, 0.008, size=len(augmented))
        augmented = augmented + noise
        
    # 2. Random phase shift (circular translation up to 30 bins)
    if np.random.rand() < 0.5:
        shift = np.random.randint(-30, 31)
        augmented = np.roll(augmented, shift)
        
    # 3. Multiplicative flux scaling (±2%)
    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.98, 1.02)
        augmented = augmented * scale
        
    return augmented


def preprocess_light_curve(time, flux, period, epoch, duration, is_training=False):
    # Runs the entire preprocessing pipeline for a single target:
    # 1. Outlier removal (5-sigma clipping)
    # 2. Phase folding
    # 3. Construct Global view (1001 bins in [-0.5, 0.5])
    # 4. Construct Local view (1001 bins in [-W, W], where W = 2.5 * duration / period)
    # 5. Clean and normalize both views (median-subtraction and std-normalization)
    # 6. (Optional) Apply data augmentation during training
    # 1. Outlier removal
    clean_time, clean_flux = remove_outliers(time, flux, sigma=5.0)
    
    # 2. Phase folding
    phase = phase_fold(clean_time, period, epoch)
    
    # 3. Global view representation
    global_view = bin_light_curve(phase, clean_flux, num_bins=1001, min_phase=-0.5, max_phase=0.5)
    global_view = normalize_view(global_view)
    
    # 4. Local view representation
    # Ensure period and duration are valid positive numbers to avoid division by zero
    if period > 0 and duration > 0:
        # Convert duration from hours to days to match period units
        duration_days = duration / 24.0
        width = 2.5 * (duration_days / period)
        # Prevent width from exceeding the phase boundary (0.5)
        width = min(width, 0.5)
    else:
        # Fallback if catalog values are missing or invalid
        width = 0.05
        
    local_view = bin_light_curve(phase, clean_flux, num_bins=1001, min_phase=-width, max_phase=width)
    local_view = normalize_view(local_view)
    
    # 5. Apply augmentations if training
    if is_training:
        global_view = augment_view(global_view)
        local_view = augment_view(local_view)
        
    return global_view, local_view
