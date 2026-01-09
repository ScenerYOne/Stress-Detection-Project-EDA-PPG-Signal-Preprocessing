# An AI-driven Stress Detection System using Physiological Signals 
This is project Stress Detection Project: EDA & PPG Signal Preprocessing with NeuroKit2
A complete guide to preprocessing Electrodermal Activity (EDA) and Photoplethysmography (PPG) signals using Python and NeuroKit2, followed by detailed documentation of all available features for stress classification research.

After completing the preprocessing steps, proceed to model training:  
 [ScenerYOne/Stress_Classification_TensorflowLSTM](https://github.com/ScenerYOne/Stress_Classification_TensorflowLSTM)


## Installation and Setup

```bash
# Install required packages
pip install -r requirements.txt

# Unzip raw data (if not already done)
unzip Raw.zip
```

### Recommended `requirements.txt`
```
neurokit2
pandas
numpy
scipy
matplotlib
seaborn
```

## Project Structure
```
preprocess-eda-ppg/
├── ppg-process.py       # PPG signal processing script
├── eda-process.py       # EDA signal processing script
├── concat-ppg.py        # Concatenate processed PPG data
├── concat-eda.py        # Concatenate processed EDA data
├── label.json           # Subject metadata (e.g., stress labels, age, etc.)
├── Raw/                 # Raw signal data
│   ├── ppg/             # Raw PPG files (s01_PPG.csv, ...)
│   └── eda/             # Raw EDA files (S01_EDA.csv, ...)
├── Processed/           # (Auto-created) Individual processed files
├── Final/               # (Auto-created) Concatenated datasets
├── requirements.txt
└── test.py              # Optional testing script
```

## File Format Requirements

### PPG Files
- Naming: `s##_PPG.csv` (e.g., `s01_PPG.csv`)
- Required columns:
  - `LocalTimestamp`: Unix timestamp (seconds)
  - `PG`: Raw PPG signal values
  - `Quality`: Signal quality flag (optional, for filtering)

### EDA Files
- Naming: `S##_EDA.csv` (e.g., `S01_EDA.csv`)
- Required columns:
  - `LocalTimestamp`: Unix timestamp (seconds)
  - `EDA`: Raw EDA signal values (microsiemens)

## Processing Pipeline

### 1. Process PPG Signals
```bash
python ppg-process.py
```
- Loads each raw PPG file
- Resamples to consistent sampling rate (e.g., 64 Hz)
- Filters low-quality segments (if Quality column exists)
- Cleans signal with `nk.ppg_clean()`
- Detects peaks and extracts features using `nk.ppg_process()`
- Saves cleaned signals and features per subject in `Processed/ppg/`

### 2. Process EDA Signals
```bash
python eda-process.py
```
- Loads each raw EDA file
- Cleans signal with `nk.eda_clean()`
- Decomposes into tonic/phasic components using `nk.eda_phasic()`
- Detects SCR peaks and extracts features
- Saves cleaned signals and features per subject in `Processed/eda/`

### 3. Concatenate Processed Data
```bash
python concat-ppg.py
python concat-eda.py
```
- Combines all processed PPG files → `Final/ppg_combined.csv`
- Combines all processed EDA files → `Final/eda_combined.csv`
- Aligns timestamps and merges with `label.json` for stress labeling

The files in `Final/` are ready for feature extraction and model input.

## Next Step: Model Training

Once preprocessing is complete, use the concatenated datasets for training:

**Stress Classification with TensorFlow LSTM**  
Repository:  [ScenerYOne/Stress_Classification_TensorflowLSTM](https://github.com/ScenerYOne/Stress_Classification_TensorflowLSTM)

Includes feature extraction, sequence preparation, LSTM model architecture, training, and evaluation.

---

## NeuroKit2 EDA Features (Complete List)

Updated for NeuroKit2 version 0.2.13

### From `nk.eda_process()`
Returns `signals` (DataFrame) and `info` (dict).

#### Columns in `signals` DataFrame
| Column               | Description                                                              |
|----------------------|--------------------------------------------------------------------------|
| EDA_Raw             | Original raw EDA signal                                                  |
| EDA_Clean           | Cleaned EDA signal                                                       |
| EDA_Tonic           | Tonic component (slow-changing Skin Conductance Level - SCL)             |
| EDA_Phasic          | Phasic component (fast-changing Skin Conductance Responses - SCR)        |
| SCR_Onsets          | Binary markers (1 at SCR onset samples)                                  |
| SCR_Peaks           | Binary markers (1 at SCR peak samples)                                   |
| SCR_Height          | SCR height including tonic                                               |
| SCR_Amplitude       | SCR amplitude excluding tonic                                            |
| SCR_RiseTime        | Rise time from onset to peak (seconds)                                   |
| SCR_Recovery        | Binary markers (1 at half-recovery points)                               |

#### Keys in `info` dict
- `SCR_Amplitude`: List of amplitudes for each SCR
- `SCR_Onsets`: List of onset indices
- `SCR_Peaks`: List of peak indices
- Sampling rate

### From `nk.eda_analyze()`

#### Event-Related Analysis (short epochs)
| Column                      | Description                                                              |
|-----------------------------|--------------------------------------------------------------------------|
| Label / Condition           | Epoch identifier                                                         |
| EDA_SCR                     | 1 if SCR present, 0 otherwise                                            |
| EDA_Peak_Amplitude          | Maximum phasic amplitude in epoch                                         |
| SCR_Peak_Amplitude          | Amplitude of first SCR                                                   |
| SCR_Peak_Amplitude_Time     | Time to peak from epoch onset                                            |
| SCR_RiseTime                | Rise time of first SCR                                                   |
| SCR_RecoveryTime            | Half-recovery time of first SCR                                          |

#### Interval-Related Analysis (long recordings)
| Column                        | Description                                                              | Condition            |
|-------------------------------|--------------------------------------------------------------------------|----------------------|
| SCR_Peaks_N                   | Total number of SCR peaks                                                | Always               |
| SCR_Peaks_Amplitude_Mean      | Mean amplitude of SCR peaks                                              | Always               |
| EDA_Tonic_SD                  | SD of tonic component                                                    | Always               |
| EDA_Sympathetic               | Power in 0.045–0.25 Hz band (sympathetic)                                 | >64 s                |
| EDA_SympatheticN              | Normalized sympathetic power                                             | >64 s                |
| EDA_Autocorrelation           | Autocorrelation at 4-second lag                                          | >30 s                |

Reference: https://neuropsychology.github.io/NeuroKit/functions/eda.html

---

## NeuroKit2 PPG Features (Complete List)

Updated for NeuroKit2 version 0.2.13

### From `nk.ppg_process()`
Returns `signals` (DataFrame) and `info` (dict).

#### Columns in `signals` DataFrame
| Column               | Description                                                              |
|----------------------|--------------------------------------------------------------------------|
| PPG_Raw             | Original raw PPG signal                                                  |
| PPG_Clean           | Cleaned PPG signal                                                       |
| PPG_Rate            | Instantaneous heart rate (bpm) from inter-peak intervals                 |
| PPG_Peaks           | Binary markers (1 at systolic peak locations)                            |

#### Keys in `info` dict
| Key                  | Description                                                              |
|----------------------|--------------------------------------------------------------------------|
| sampling_rate       | Sampling rate (Hz)                                                       |
| PPG_Peaks           | Indices of systolic peaks                                                |

### From `nk.ppg_analyze()`

#### Event-Related Analysis (short epochs)
| Column                          | Description                                                              |
|---------------------------------|--------------------------------------------------------------------------|
| Label / Index                   | Epoch identifier                                                         |
| PPG_Rate_Baseline               | Pre-event heart rate                                                     |
| PPG_Rate_Max                    | Maximum heart rate in epoch                                              |
| PPG_Rate_Min                    | Minimum heart rate in epoch                                              |
| PPG_Rate_Mean                   | Mean heart rate in epoch                                                 |
| PPG_Rate_SD                     | SD of heart rate in epoch                                                |
| PPG_Rate_Max_Time               | Time to max HR from onset                                                |
| PPG_Rate_Min_Time               | Time to min HR from onset                                                |
| PPG_Rate_Trend_Linear           | Linear trend coefficient (experimental)                                  |
| PPG_Rate_Trend_Quadratic        | Quadratic trend coefficient (experimental)                               |
| PPG_Rate_Trend_R2               | R² of quadratic fit (experimental)                                       |

#### Interval-Related Analysis (long recordings)
| Column                        | Description                                                              |
|-------------------------------|--------------------------------------------------------------------------|
| PPG_Rate_Mean                 | Overall mean heart rate                                                  |
| HRV_* (90+ features)          | Full Heart Rate Variability metrics                                      |

**Selected HRV examples:**
- Time-domain: `HRV_MeanNN`, `HRV_SDNN`, `HRV_RMSSD`, `HRV_pNN50`
- Frequency-domain: `HRV_LF`, `HRV_HF`, `HRV_LFHF`, `HRV_VLF`, `HRV_TP`
- Nonlinear: `HRV_SampEn`, `HRV_ApEn`, `HRV_HFD`, `HRV_LZC`

**Note:** Only systolic peaks are detected (no dicrotic notch detection).

Reference: https://neuropsychology.github.io/NeuroKit/functions/ppg.html

---




