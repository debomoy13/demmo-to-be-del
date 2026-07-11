import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from preprocess import preprocess_light_curve, augment_view

class ExoplanetDataset(Dataset):
    # PyTorch Dataset that loads target metadata, stellar catalog parameters,
    # and light curve files. Handles missing files by generating zero-vector inputs.
    def __init__(self, df_meta, dataset_dir, augment=False):
        self.df_meta = df_meta.reset_index(drop=True)
        self.dataset_dir = dataset_dir
        self.augment = augment
        
        # Catalog features to extract:
        # 1. Orbital period P (koi_period)
        # 2. Transit duration \Delta t (koi_duration)
        # 3. Transit depth \delta (koi_depth)
        # 4. Planet radius Rp (koi_prad)
        # 5. Equilibrium temperature Teq (koi_teq)
        # 6. Stellar effective temperature Teff (koi_steff)
        # 7. Stellar surface gravity log g (koi_slogg)
        # 8. Stellar metallicity [Fe/H] (koi_smet)
        self.stellar_cols = [
            'koi_period',
            'koi_duration',
            'koi_depth',
            'koi_prad',
            'koi_teq',
            'koi_steff',
            'koi_slogg',
            'koi_smet'
        ]
        # Calculate medians for filling NaN catalog entries
        self.stellar_medians = self.df_meta[self.stellar_cols].median()
        
        # Label mapping matching the standard configuration
        self.label_mapping = {
            'transit': 0,
            'stellar_eclipse': 1,
            'not_transit': 2,
            'centroid_offset': 3,
            'Ephemeris match': 4,
            'unlabeled': 2
        }
        self.cache = {}

    def __len__(self):
        return len(self.df_meta)

    def __getitem__(self, idx):
        row = self.df_meta.iloc[idx]
        kepid = int(row["kepid"])
        
        # 1. Label classification target
        label_str = row.get("signal_class", "unlabeled")
        class_idx = self.label_mapping.get(label_str, 2)
        y_class = torch.tensor(class_idx, dtype=torch.long)
        
        # 2. Transit confidence target (1.0 for true transits, 0.0 for false positives/other)
        confidence = 1.0 if class_idx == 0 else 0.0
        y_confidence = torch.tensor([confidence], dtype=torch.float32)
        
        # 3. Retrieve folding parameters
        period = float(row["koi_period"]) if not pd.isna(row["koi_period"]) else 0.0
        epoch = float(row["koi_time0bk"]) if not pd.isna(row["koi_time0bk"]) else 0.0
        duration = float(row["koi_duration"]) if not pd.isna(row["koi_duration"]) else 0.0
        
        # 4. Extract catalog features and fill NaNs
        stellar_vals = []
        for col in self.stellar_cols:
            val = row[col]
            if pd.isna(val):
                stellar_vals.append(float(self.stellar_medians[col]))
            else:
                stellar_vals.append(float(val))
        x_stellar = torch.tensor(stellar_vals, dtype=torch.float32)
        
        # 5. Load and preprocess light curve
        if idx in self.cache:
            global_view, local_view = self.cache[idx]
        else:
            file_path = os.path.join(self.dataset_dir, f"{kepid}.npz")
            if os.path.exists(file_path):
                data = np.load(file_path)
                time = data["time"]
                flux = data["flux"]            
                # Perform phase folding, binning, and normalization (without augmentation)
                global_view, local_view = preprocess_light_curve(
                    time=time,
                    flux=flux,
                    period=period,
                    epoch=epoch,
                    duration=duration,
                    is_training=False
                )
            else:
                # Handle missing files by setting both representations to zero vectors
                global_view = np.zeros(1001, dtype=np.float32)
                local_view = np.zeros(1001, dtype=np.float32)
            self.cache[idx] = (global_view, local_view)
            
        # Apply augmentations if training
        if self.augment:
            global_view = augment_view(global_view)
            local_view = augment_view(local_view)
            
        # Convert views to tensors and add channel dimension: shape (1, 1001)
        x_global = torch.tensor(global_view, dtype=torch.float32).unsqueeze(0)
        x_local = torch.tensor(local_view, dtype=torch.float32).unsqueeze(0)
        
        # 6. Scaled regression targets for multi-task learning
        y_regression = torch.tensor([
            np.log1p(max(0.0, float(row["koi_depth"]))) / 10.0 if not pd.isna(row["koi_depth"]) else 0.0,
            duration / 10.0,
            np.log1p(max(0.0, period)) / 5.0,
            np.log1p(abs(epoch)) / 10.0
        ], dtype=torch.float32)
        
        return x_global, x_local, x_stellar, y_class, y_regression, y_confidence


def get_data_loaders(csv_path, dataset_dir, batch_size=32, test_size=0.15, val_size=0.15, seed=42):
    # Reads metadata, splits it into stratified train, validation, and test datasets,
    # and returns PyTorch DataLoader objects.
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")
        
    df_meta = pd.read_csv(csv_path, comment='#')
    
    # Ensure label mappings exist to perform stratified train_test_splits
    label_mapping = {
        'transit': 0,
        'stellar_eclipse': 1,
        'not_transit': 2,
        'centroid_offset': 3,
        'Ephemeris match': 4,
        'unlabeled': 2
    }
    df_meta["label_idx"] = df_meta["signal_class"].apply(lambda x: label_mapping.get(x, 2))
    
    # 1. Stratified split for test set
    train_val_df, test_df = train_test_split(
        df_meta, 
        test_size=test_size, 
        random_state=seed, 
        stratify=df_meta["label_idx"]
    )
    
    # 2. Stratified split for validation set
    rel_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=rel_val_size, 
        random_state=seed, 
        stratify=train_val_df["label_idx"]
    )
    
    print(f"Dataset split counts: Train = {len(train_df)}, Val = {len(val_df)}, Test = {len(test_df)}")
    
    # Instantiate PyTorch datasets
    train_ds = ExoplanetDataset(train_df, dataset_dir, augment=True)
    val_ds = ExoplanetDataset(val_df, dataset_dir, augment=False)
    test_ds = ExoplanetDataset(test_df, dataset_dir, augment=False)
    
    # Construct standard DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, test_loader
