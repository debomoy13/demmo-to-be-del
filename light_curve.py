# Downloader script for Kepler KOI light curves to build a balanced dataset of 7,585 samples
import os
import pandas as pd
import numpy as np
from lightkurve import search_lightcurve
import time

# Load labeled cumulative KOI catalog
csv_path = "modified datasets/koi_cumulative_labeled.csv"
if not os.path.exists(csv_path):
    # Fallback to root directory if modified datasets folder is missing
    csv_path = "koi_cumulative_labeled.csv"

print(f"Reading KOI catalog from {csv_path}...")
koi = pd.read_csv(csv_path, comment='#')

# Target distribution to reach 7,585 samples (matching paper size)
# and distributing samples as balanced as possible across classes
target_counts = {
    'transit': 2746,
    'stellar_eclipse': 2210,
    'centroid_offset': 1206,
    'not_transit': 1204,
    'Ephemeris match': 124,
    'unlabeled': 95
}

# Create reproducible sample subset
selected_dfs = []
for label, target in target_counts.items():
    df_class = koi[koi["signal_class"] == label]
    if len(df_class) <= target:
        selected_dfs.append(df_class)
    else:
        selected_dfs.append(df_class.sample(n=target, random_state=42))

subset = pd.concat(selected_dfs).sample(frac=1.0, random_state=42)  # Shuffle classes
os.makedirs("dataset", exist_ok=True)

records = []
total_subset = len(subset)
print(f"Targeting a total of {total_subset} light curves for download/indexing...")

skipped_count = 0
downloaded_count = 0
failed_count = 0

for i, (_, row) in enumerate(subset.iterrows(), 1):
    kepid = int(row["kepid"])
    label = row["signal_class"]
    filepath = f"dataset/{kepid}.npz"
    
    # 1. Check if file is already downloaded and is valid
    is_valid = False
    if os.path.exists(filepath):
        try:
            data = np.load(filepath)
            if "flux" in data and "time" in data:
                is_valid = True
        except Exception:
            # File might be corrupted, delete it to re-download
            try:
                os.remove(filepath)
            except Exception:
                pass

    if is_valid:
        records.append({
            "kepid": kepid,
            "label": label,
            "file": filepath
        })
        skipped_count += 1
        if i % 100 == 0 or i == total_subset:
            print(f"Progress: Checked {i}/{total_subset} - Skipped {skipped_count} existing, Downloaded {downloaded_count}, Failed {failed_count}")
        continue

    # 2. Download and preprocess the light curve
    retries = 3
    success = False
    for attempt in range(1, retries + 1):
        try:
            # search and download light curve
            lc = search_lightcurve(f"KIC {kepid}", mission="kepler").download()
            if lc is not None:
                lc = lc.remove_nans()
                lc = lc.normalize()
                
                # Save as NPZ
                np.savez(
                    filepath,
                    time=lc.time.value,
                    flux=lc.flux.value
                )
                records.append({
                    "kepid": kepid,
                    "label": label,
                    "file": filepath
                })
                downloaded_count += 1
                success = True
                print(f"[{i}/{total_subset}] Successfully downloaded and saved KIC {kepid} ({label})")
                break
            else:
                print(f"Warning (Attempt {attempt}/{retries}): Lightcurve search returned None for KIC {kepid}")
        except Exception as e:
            print(f"Warning (Attempt {attempt}/{retries}) for KIC {kepid}: {e}")
            time.sleep(2)  # Wait before retry
            
    if not success:
        failed_count += 1
        print(f"Error: Failed to download KIC {kepid} after {retries} attempts. Skipping.")

# Save updated dataset index CSV
pd.DataFrame(records).to_csv("dataset_index.csv", index=False)
print("=" * 60)
print("DOWNLOAD AND VERIFICATION SUITE COMPLETE")
print(f"Total target list: {total_subset}")
print(f"Successfully indexed: {len(records)}")
print(f"Skipped (already exist): {skipped_count}")
print(f"Newly downloaded: {downloaded_count}")
print(f"Failed downloads: {failed_count}")
print("=" * 60)