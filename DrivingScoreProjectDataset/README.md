# Driving Behaviour Dataset
### Two-Wheeler Motion Sensor Data — Driving Risk Score Project

Dataset collected manually using an Android sensor application built by seniors at our institute. The app captures real-time accelerometer and gyroscope readings while riding a two-wheeler and exports them as CSV files.

---

## Data Collection App

> **App:** [Two-Wheeler Road Surface Classifier](https://github.com/Harshit-Soni78/Two-Wheeler-Road-Surface-Classifier)
> Built by Harshit Soni — captures accelerometer and gyroscope data from an Android phone mounted on a two-wheeler.

The app records 6 sensor channels at ~10ms intervals and saves each session as a CSV file that can be exported to a computer.

---

## CSV Format

Each recorded session file has the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `ID` | string | Rider name (dropped during processing) |
| `SrNo` | int | Row serial number |
| `Timestamp` | long | Unix timestamp in milliseconds |
| `X_Acc` | float | Accelerometer X-axis (m/s²) |
| `Y_Acc` | float | Accelerometer Y-axis (m/s²) |
| `Z_Acc` | float | Accelerometer Z-axis (m/s²) |
| `X_Gyro` | float | Gyroscope X-axis (deg/s) |
| `Y_Gyro` | float | Gyroscope Y-axis (deg/s) |
| `Z_Gyro` | float | Gyroscope Z-axis (deg/s) |

Sample row:
```
ID,       SrNo, Timestamp,      X_Acc,    Y_Acc,   Z_Acc,    X_Gyro,    Y_Gyro,  Z_Gyro
Ayan,     1,    1739352388986,  -0.79173, -4.23709, 9.1687,  -4.71483,  6.08045, 12.31348
```

---

## Rating System

Sessions are manually labelled by placing them into star-rated folders. The rating reflects the **driving behaviour observed during that session**:

| Folder | Rating | Driving Style | Description |
|--------|--------|---------------|-------------|
| `1 star/` | 1 | Dangerous | Highly aggressive — sudden braking, sharp turns, overspeeding |
| `2 star/` | 2 | Rush | Fast and impatient — frequent acceleration/braking events |
| `3 star/` | 3 | Moderate | Average driving — mix of smooth and some aggressive moments |
| `4 star/` | 4 | Careful | Mostly smooth — controlled speed, gentle braking |
| `5 star/` | 5 | Safe | Perfectly smooth — consistent speed, no sudden events |

---

## Dataset Statistics

| Property | Value |
|---|---|
| Total raw rows | 2,987,342 |
| Total session files | 193 |
| After cleaning | 2,752,364 rows (92.13% retention) |
| Total windows (100 rows, 50% overlap) | 54,850 |
| Train / Test split | 43,880 / 10,970 (80/20 stratified) |
| Sampling rate | ~100 Hz (~10ms per row) |
| Window duration | ~1–2 seconds per window |

### Distribution across ratings

| Rating | Raw rows | Windows | % of dataset |
|--------|----------|---------|--------------|
| 1 star (dangerous) | 574,900 | 21,722 | 39.6% |
| 2 star (rush) | 486,181 | 7,722 | 14.1% |
| 3 star (moderate) | 705,882 | 9,816 | 17.9% |
| 4 star (careful) | 790,282 | 10,390 | 18.9% |
| 5 star (safe) | 430,097 | 5,200 | 9.5% |

---

## Dataset Variations

The dataset covers all required variations for a robust model:

### Riders
Ayan, Rohan, Sanjay, Vikram, Yogesh, Manish, Sonu, Shabbir, Lalit, Hitesh, Jatin, Chahit, Tanishq, Lokesh, Mahesh, Vishal, Vivvan

### Vehicles
- Scooter (Activa and similar)
- Motorcycle (Pulsar and similar)

### Speed Ranges
- Slow (0–20 km/h)
- Medium (20–40 km/h)
- Fast (40+ km/h)

### Phone Placements
- Pocket
- Handlebar mount
- Backpack
- Tank mount

---

## Folder Structure

```
data/raw_real/
├── 1 star/
│   ├── Ayan_smooth_1(1).csv
│   ├── sonu_dangerous01.csv
│   ├── vikram_1.csv
│   └── ... (31 files)
├── 2 star/
│   └── ... (36 files)
├── 3 star/
│   └── ... (50 files)
├── 4 star/
│   └── ... (48 files)
└── 5 star/
    └── ... (28 files)
```

---

## How to Use This Dataset

### With the Driving Risk Score pipeline

```bash
# Clone the project
git clone https://github.com/YourUsername/NNLAB.git
cd NNLAB/DrivingScoreProject

# Place this dataset folder at:
# DrivingScoreProject/data/raw_real/

# Run the full pipeline
python run_pipeline.py
```

The pipeline will automatically:
1. Read all CSVs from star folders and assign ratings
2. Drop the `ID` column
3. Clean bad rows (zero sensors, duplicates, frozen values)
4. Build 100-row sliding windows
5. Train LSTM / GRU / Transformer models

### Manually with pandas

```python
import pandas as pd

# Load one session
df = pd.read_csv('1 star/sonu_dangerous01.csv')

# Drop rider name column
df = df.drop(columns=['ID'])

# Add rating from folder name
df['Rating'] = 1

print(df.head())
print(f"Shape: {df.shape}")
```

---

## Data Collection Process

1. Installed the Android app from the seniors' repository on multiple phones
2. Each rider mounted the phone in the designated placement (pocket / handlebar / backpack)
3. Rode the two-wheeler under natural conditions — no simulated or artificial driving
4. Exported CSV after each session
5. Manually labelled each session by placing it in the appropriate star folder based on observed driving behaviour
6. Repeated across multiple riders, vehicles, speeds, and placements

---

## Notes

- Some session files have very few rows (under 200) — these are valid but produce fewer windows
- File naming is inconsistent across riders — the pipeline handles this automatically
- One file (`Lokesh_safe1.csv`) was skipped during processing due to missing sensor columns
- `Z_Acc` should never be zero on a real device (gravity = 9.8 m/s²) — rows with `Z_Acc = 0` are cleaned automatically

---

## Related Links

| Resource | Link |
|---|---|
| Data collection app | [Two-Wheeler Road Surface Classifier](https://github.com/Harshit-Soni78/Two-Wheeler-Road-Surface-Classifier) |
| Driving Risk Score project | [DrivingScoreProject](../DrivingScoreProject/README.md) |