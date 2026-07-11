import pandas as pd

koi = pd.read_csv("cumulative_2026.06.24_09.53.31.csv" , comment='#')

def assign_label(row):

    if row["koi_disposition"] in ("CONFIRMED", "CANDIDATE"):
        return "transit"
    elif row["koi_fpflag_ss"] == 1:
        return "stellar_eclipse"
    elif row["koi_fpflag_co"] == 1:
        return "centroid_offset"
    elif row["koi_fpflag_nt"] == 1:
        return "not_transit"
    elif row["koi_fpflag_ec"]==1:
        return "Ephemeris match"
    else:
        return "unlabeled"

koi["signal_class"] = koi.apply(assign_label, axis=1)
print(koi["signal_class"].value_counts())

koi.to_csv("koi_cumulative_labeled.csv", index=False)