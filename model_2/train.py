import os
import argparse
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

try:
    from model_2.data_load import ExoplanetDataset
    from model_2.concantecenattion import ExoplanetLateFusionModel
except ImportError:
    from data_load import ExoplanetDataset
    from concantecenattion import ExoplanetLateFusionModel

class ExoplanetMultiTaskLoss(nn.Module):
    """
    Computes joint loss combining:
    1. Inverse-frequency weighted multi-class Cross Entropy for classification.
    2. Masked Mean Squared Error (MSE) for transit regression parameters (only on true transit samples, label = 0).
    3. Binary Cross Entropy (BCE) for transit confidence scores.
    """
    def __init__(self, class_weights=None, lambda_reg=0.4, lambda_conf=1.0):
        super().__init__()
        self.class_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.reg_loss_fn = nn.MSELoss(reduction='none')
        self.conf_loss_fn = nn.BCELoss()
        self.lambda_reg = lambda_reg
        self.lambda_conf = lambda_conf

    def forward(self, class_logits, reg_outputs, confidence, targets_class, targets_reg, targets_conf):
        # 1. Classification Loss (Weighted Cross Entropy)
        loss_class = self.class_loss_fn(class_logits, targets_class)
        
        # 2. Masked Regression Loss (computed ONLY on True Transit samples where target label is 0)
        transit_mask = (targets_class == 0)
        if transit_mask.sum() > 0:
            reg_pred = reg_outputs[transit_mask]
            reg_true = targets_reg[transit_mask]
            loss_reg = self.reg_loss_fn(reg_pred, reg_true).mean()
        else:
            loss_reg = torch.tensor(0.0, device=class_logits.device)
            
        # 3. Confidence Head Loss (BCE)
        # Disable autocasting for BCELoss as it is unstable in FP16
        device_type = confidence.device.type
        if device_type in ['cuda', 'cpu']:
            with torch.amp.autocast(device_type=device_type, enabled=False):
                clamped_conf = torch.clamp(confidence.float(), min=1e-7, max=1.0 - 1e-7)
                loss_conf = self.conf_loss_fn(clamped_conf, targets_conf.float())
        else:
            clamped_conf = torch.clamp(confidence, min=1e-7, max=1.0 - 1e-7)
            loss_conf = self.conf_loss_fn(clamped_conf, targets_conf)
        
        # Total joint loss
        total_loss = loss_class + self.lambda_reg * loss_reg + self.lambda_conf * loss_conf
        
        return total_loss, loss_class, loss_reg, loss_conf


def train_model(args):
    # Ensure directories exist
    os.makedirs("model_2/checkpoints", exist_ok=True)
    os.makedirs("model_2/results", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load metadata and split into stratified splits (70/15/15)
    csv_path = args.csv_path
    if not os.path.exists(csv_path):
        # Try common alternatives
        alternatives = [
            "modified datasets/koi_cumulative_labeled.csv",
            "koi_cumulative_labeled.csv",
            "dataset_index.csv"
        ]
        found = False
        for alt in alternatives:
            if os.path.exists(alt):
                csv_path = alt
                print(f"Metadata file '{args.csv_path}' not found. Using alternative: '{alt}'")
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Metadata file not found: {args.csv_path}")
            
    print(f"Loading metadata from {csv_path}...")
    df_meta = pd.read_csv(csv_path, comment='#')
    
    # Verify if signal_class column is present. If not (e.g. if loaded dataset_index.csv directly), find alternative
    if "signal_class" not in df_meta.columns:
        alternatives = [
            "modified datasets/koi_cumulative_labeled.csv",
            "koi_cumulative_labeled.csv"
        ]
        found = False
        for alt in alternatives:
            if os.path.exists(alt):
                temp_df = pd.read_csv(alt, comment='#')
                if "signal_class" in temp_df.columns:
                    df_meta = temp_df
                    csv_path = alt
                    print(f"Loaded alternative metadata file containing 'signal_class': {alt}")
                    found = True
                    break
        if not found:
            raise KeyError("Column 'signal_class' not found in metadata file and no valid alternative labeled catalog was found.")
    
    # Class labels mapping index
    label_mapping = {
        'transit': 0,
        'stellar_eclipse': 1,
        'not_transit': 2,
        'centroid_offset': 3,
        'Ephemeris match': 4,
        'unlabeled': 2
    }
    df_meta["label_idx"] = df_meta["signal_class"].apply(lambda x: label_mapping.get(x, 2))
    
    # Stratified split: Train (70%), Val (15%), Test (15%)
    # First split out test set (15%)
    train_val_df, test_df = train_test_split(
        df_meta,
        test_size=0.15,
        random_state=42,
        stratify=df_meta["label_idx"]
    )
    # Split remaining 85% into train (70% of total) and validation (15% of total)
    val_rel_size = 0.15 / 0.85
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_rel_size,
        random_state=42,
        stratify=train_val_df["label_idx"]
    )
    
    print(f"Data Splits: Train = {len(train_df)} samples, Val = {len(val_df)} samples, Test = {len(test_df)} samples")
    
    # 2. Compute Inverse Class Frequencies to address imbalance
    class_counts = train_df["label_idx"].value_counts().sort_index()
    num_classes = 5
    counts = [class_counts.get(i, 1) for i in range(num_classes)]
    
    # Calculate inverse frequencies
    inv_freq = [1.0 / c for c in counts]
    class_weights = torch.tensor(inv_freq, dtype=torch.float32).to(device)
    
    print(f"Training Class Counts: {dict(class_counts)}")
    print(f"Calculated Inverse Class Weights: {inv_freq}")
    
    # 3. Create sampler and dataloaders
    train_ds = ExoplanetDataset(train_df, args.dataset_dir, augment=True)
    val_ds = ExoplanetDataset(val_df, args.dataset_dir, augment=False)
    test_ds = ExoplanetDataset(test_df, args.dataset_dir, augment=False)
    
    # Weighted Random Sampler to balance training batches
    train_labels = train_df["label_idx"].values
    sample_weights = [inv_freq[label] * (args.ephemeris_boost if label == 4 else 1.0) for label in train_labels]
    print(f"Applied 'Ephemeris match' class (label 4) sampling boost: {args.ephemeris_boost}x")
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )
    
    # 4. Instantiate Model, Optimizer, and OneCycleLR Scheduler
    model = ExoplanetLateFusionModel(num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    total_steps = len(train_loader) * args.epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3 * args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.15,
        anneal_strategy='cos'
    )
    
    # Joint Loss Criterion
    criterion = ExoplanetMultiTaskLoss(class_weights=class_weights)
    
    # Setup mixed-precision scaling
    use_amp = (device.type == 'cuda') and args.use_amp
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    # 5. Training loop with early stopping on validation ROC AUC
    best_val_auc = -1.0
    patience_counter = 0
    history = []
    
    print("\nStarting model training...")
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # --- Training Epoch ---
        model.train()
        train_loss = 0.0
        train_class_loss = 0.0
        train_reg_loss = 0.0
        train_conf_loss = 0.0
        
        steps = 0
        for x_global, x_local, x_stellar, y_class, y_reg, y_conf in train_loader:
            x_global = x_global.to(device)
            x_local = x_local.to(device)
            x_stellar = x_stellar.to(device)
            y_class = y_class.to(device)
            y_reg = y_reg.to(device)
            y_conf = y_conf.to(device)
            
            optimizer.zero_grad()
            
            # Autocast FP16 forward pass
            if use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    class_logits, reg_outputs, confidence, _, _ = model(x_global, x_local, x_stellar)
                    loss, loss_cls, loss_rg, loss_cf = criterion(
                        class_logits, reg_outputs, confidence, y_class, y_reg, y_conf
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                class_logits, reg_outputs, confidence, _, _ = model(x_global, x_local, x_stellar)
                loss, loss_cls, loss_rg, loss_cf = criterion(
                    class_logits, reg_outputs, confidence, y_class, y_reg, y_conf
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                
            scheduler.step()
            
            train_loss += loss.item()
            train_class_loss += loss_cls.item()
            train_reg_loss += loss_rg.item()
            train_conf_loss += loss_cf.item()
            
            steps += 1
            if args.dry_run and steps >= 2:
                break
                
        # Average epoch training metrics
        train_loss /= steps
        train_class_loss /= steps
        train_reg_loss /= steps
        train_conf_loss /= steps
        
        # --- Validation Epoch ---
        model.eval()
        val_loss = 0.0
        val_class_loss = 0.0
        val_reg_loss = 0.0
        val_conf_loss = 0.0
        
        all_val_labels = []
        all_val_probs = []
        all_val_preds = []
        
        val_steps = 0
        with torch.no_grad():
            for x_global, x_local, x_stellar, y_class, y_reg, y_conf in val_loader:
                x_global = x_global.to(device)
                x_local = x_local.to(device)
                x_stellar = x_stellar.to(device)
                y_class = y_class.to(device)
                y_reg = y_reg.to(device)
                y_conf = y_conf.to(device)
                
                # Autocast evaluation
                if use_amp:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        class_logits, reg_outputs, confidence, _, _ = model(x_global, x_local, x_stellar)
                        loss, loss_cls, loss_rg, loss_cf = criterion(
                            class_logits, reg_outputs, confidence, y_class, y_reg, y_conf
                        )
                else:
                    class_logits, reg_outputs, confidence, _, _ = model(x_global, x_local, x_stellar)
                    loss, loss_cls, loss_rg, loss_cf = criterion(
                        class_logits, reg_outputs, confidence, y_class, y_reg, y_conf
                    )
                    
                val_loss += loss.item()
                val_class_loss += loss_cls.item()
                val_reg_loss += loss_rg.item()
                val_conf_loss += loss_cf.item()
                
                # Get probabilities
                probs = model.get_probabilities(class_logits)
                
                all_val_labels.extend(y_class.cpu().numpy())
                all_val_probs.extend(probs.cpu().numpy())
                all_val_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
                
                val_steps += 1
                if args.dry_run and val_steps >= 2:
                    break
                    
        # Average epoch validation metrics
        val_loss /= val_steps
        val_class_loss /= val_steps
        val_reg_loss /= val_steps
        val_conf_loss /= val_steps
        
        # Calculate Validation Metrics
        all_val_labels = np.array(all_val_labels)
        all_val_probs = np.array(all_val_probs)
        all_val_preds = np.array(all_val_preds)
        
        val_acc = accuracy_score(all_val_labels, all_val_preds)
        
        # Calculate multi-class OvR ROC AUC
        try:
            val_auc = roc_auc_score(
                all_val_labels,
                all_val_probs,
                multi_class='ovr',
                labels=np.arange(num_classes)
            )
        except Exception:
            val_auc = 0.5  # Fallback for dry runs / edge cases
            
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Time: {epoch_time:.1f}s | "
              f"Loss: {train_loss:.4f} (Val: {val_loss:.4f}) | "
              f"Val Acc: {val_acc:.4f} | Val ROC AUC: {val_auc:.4f}")
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_class_loss": train_class_loss,
            "train_reg_loss": train_reg_loss,
            "train_conf_loss": train_conf_loss,
            "val_loss": val_loss,
            "val_class_loss": val_class_loss,
            "val_reg_loss": val_reg_loss,
            "val_conf_loss": val_conf_loss,
            "val_accuracy": val_acc,
            "val_roc_auc": val_auc
        })
        
        # Save validation metrics log
        with open("model_2/results/training_history.json", "w") as f:
            json.dump(history, f, indent=4)
            
        # Early Stopping check on Validation ROC AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), "model_2/checkpoints/best_model.pth")
            print(f"==> Validation AUC improved to {best_val_auc:.4f}. Saved best model checkpoint.")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly Stopping triggered: Val ROC AUC did not improve for {args.patience} epochs.")
                break
                
        if args.dry_run and epoch >= 1:
            print("\nDry run completed successfully.")
            break
            
    print(f"\nTraining completed! Best Validation ROC AUC: {best_val_auc:.4f}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Exoplanet Late Fusion Deep Learning Model")
    parser.add_argument("--csv_path", type=str, default="modified datasets/koi_cumulative_labeled.csv", help="Path to index dataset CSV file")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Directory containing the npz target files")
    parser.add_argument("--batch_size", type=int, default=32, help="DataLoader batch size")
    parser.add_argument("--epochs", type=int, default=80, help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak optimizer learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay parameter")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping epoch patience limit")
    parser.add_argument("--dry_run", action="store_true", help="Perform a fast shape/pipeline dry run")
    parser.add_argument("--use_amp", action="store_true", help="Enable automatic mixed precision (AMP) / mixed precision training (might cause NaN issues with raw features)")
    parser.add_argument("--ephemeris_boost", type=float, default=2.0, help="Multiplicative boost factor for 'Ephemeris match' class sampling weight")
    
    args = parser.parse_args()
    
    train_model(args)
