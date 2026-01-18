import pandas as pd
import numpy as np
import neurokit2 as nk
import glob
from pathlib import Path
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import traceback
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial


def load_json(file_path: Path) -> dict:
    """Load JSON file"""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data.get('label', [])


# Constants
SAMPLING_RATE = 100
WINDOW_SIZE = 300  # 5 minutes (300 seconds)
WINDOW_SHIFT = 1  # Shift by 5 seconds for each window (increased from 1 for speed)
WINDOW_SAMPLES = WINDOW_SIZE * SAMPLING_RATE  
RAW_PATH = Path('Raw/ppg')
PROCESSED_PATH = Path('Processed/ppg')
ENABLE_PLOTS = False  
Label = load_json("label.json")


def process_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Process timestamp values correctly"""
    if 'LocalTimestamp' not in df.columns:
        print(f"Warning: 'LocalTimestamp' column missing in data")
        return df

    if 'DateTime' not in df.columns:
        sample_timestamp = df['LocalTimestamp'].iloc[0]
        print(f"Original timestamp: {sample_timestamp}")

        df['DateTime'] = pd.to_datetime(df['LocalTimestamp'], unit='s')
        df['DateTime'] = df['DateTime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Bangkok').dt.tz_localize(None)
        print(f"Converted first timestamp: {df['DateTime'].iloc[0]}")
    
    return df

def get_subject_id_from_filename(filename: str) -> str:
    """Extract subject ID from filename"""
    return filename.split('_')[0].upper()


def get_label_for_subject(subject_id: str, labels: list) -> dict:
    """Get label for a specific subject"""
    subject_id = subject_id.upper()
    for label in labels:
        if label.get('id', '').upper() == subject_id:
            return label
    return None


def calculate_hrv_metrics(peaks, sampling_rate, minimal=False):
    """
    Calculate HRV metrics from peaks, with error handling
    If minimal=True, calculate only essential metrics for better performance
    """
    try:
        if len(peaks) < 10:
            print(f"Too few peaks ({len(peaks)}) for reliable HRV analysis")
            return {}

        if minimal:
            rr_intervals = np.diff(peaks) / sampling_rate * 1000

            hrv_dict = {
                'HRV_MeanNN': np.mean(rr_intervals),
                'HRV_SDNN': np.std(rr_intervals),
                'HR': 60000 / np.mean(rr_intervals) if len(rr_intervals) > 0 else np.nan,
                'HRV_RMSSD': np.sqrt(np.mean(np.diff(rr_intervals) ** 2)) if len(rr_intervals) > 1 else np.nan,
                'HRV_SDSD': np.std(np.diff(rr_intervals)) if len(rr_intervals) > 1 else np.nan,
                'HRV_MedianNN': np.median(rr_intervals),
                'HRV_pNN50': 100 * np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals) if len(rr_intervals) > 1 else np.nan,
                'HRV_pNN20': 100 * np.sum(np.abs(np.diff(rr_intervals)) > 20) / len(rr_intervals) if len(rr_intervals) > 1 else np.nan,
                'HRV_CVNN': (np.std(rr_intervals) / np.mean(rr_intervals)) * 100 if np.mean(rr_intervals) > 0 else np.nan,
                'HRV_CVSD': (np.std(np.diff(rr_intervals)) / np.mean(rr_intervals)) * 100 if len(rr_intervals) > 1 and np.mean(rr_intervals) > 0 else np.nan,
                'HRV_SD1': np.std(np.diff(rr_intervals) / np.sqrt(2)) if len(rr_intervals) > 1 else np.nan,
                'HRV_SD2': np.sqrt(2 * np.var(rr_intervals) - 0.5 * np.var(np.diff(rr_intervals))) if len(rr_intervals) > 1 else np.nan,
            }

            if len(rr_intervals) > 0:
                hrv_dict['HRV_MinNN'] = np.min(rr_intervals)
                hrv_dict['HRV_MaxNN'] = np.max(rr_intervals)
            
            return hrv_dict
        
        # Calculate time-domain HRV metrics
        hrv_time = nk.hrv_time(peaks, sampling_rate=sampling_rate, show=False)

        if len(peaks) >= 30:  
            try:
                hrv_freq = nk.hrv_frequency(peaks, sampling_rate=sampling_rate, show=False)
            except Exception as e:
                print(f"Warning: Frequency-domain HRV calculation failed: {e}")
                hrv_freq = pd.DataFrame(index=[0]) 
        else:
            hrv_freq = pd.DataFrame(index=[0])  

        if len(peaks) >= 20:  
            try:
                hrv_nonlinear = nk.hrv_nonlinear(peaks, sampling_rate=sampling_rate, show=False)
            except Exception as e:
                print(f"Warning: Nonlinear HRV calculation failed: {e}")
                hrv_nonlinear = pd.DataFrame(index=[0])  
        else:
            hrv_nonlinear = pd.DataFrame(index=[0])
        
        hrv_metrics = pd.concat([hrv_time, hrv_freq, hrv_nonlinear], axis=1)
        
        hrv_dict = {}
        for col in hrv_metrics.columns:
            hrv_dict[f'{col}'] = hrv_metrics[col].values[0] if not pd.isna(hrv_metrics[col].values[0]) else np.nan
        
        # Calculate heart rate
        if len(peaks) >= 2:
            rr_intervals = np.diff(peaks) / sampling_rate
            median_rr = np.median(rr_intervals)
            heart_rate = 60 / median_rr if median_rr > 0 else np.nan
            hrv_dict['HR'] = heart_rate
        
        return hrv_dict
    
    except Exception as e:
        print(f"Error calculating HRV metrics: {e}")
        traceback.print_exc()
        return {}


def process_ppg_file(file_path: Path) -> None:
    """
    Processes a single PPG file with sliding window approach for HRV feature extraction.
    Optimized for better performance, with all required columns included.
    """
    start_time = time.time()
    try:
        subject_id = get_subject_id_from_filename(file_path.name)
        subject_label = get_label_for_subject(subject_id, Label)

        if not subject_label:
            print(f"Warning: No label found for subject {subject_id}")
            return

        output_filename = f"{file_path.stem}_processed.csv" 
        output_path = PROCESSED_PATH / output_filename

        PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, 'a') as f:
                pass  
        except PermissionError:
            print(f"Permission error on {output_path}, trying alternative location")
            alt_path = Path(os.path.expanduser("~")) / "PPG_Processed"
            alt_path.mkdir(parents=True, exist_ok=True)
            output_path = alt_path / output_filename
            print(f"Using alternative path: {output_path}")

        data = pd.read_csv(file_path)
        print(f"Processing {file_path.name}, data shape: {data.shape}")

        # Process timestamp
        data = process_timestamp(data)

        # Identify PPG signal column
        ppg_columns = [col for col in data.columns if
                      any(ppg in col.lower() for ppg in ['ppg', 'pg', 'photoplethysmography', 'pulse'])]

        if ppg_columns:
            ppg_column = ppg_columns[0]
            print(f"Using column '{ppg_column}' as PPG signal")
        else:
            numeric_cols = [col for col in data.columns if data[col].dtype.kind in 'if' and col != 'DateTime']
            ppg_column = numeric_cols[0] if numeric_cols else data.columns[1]
            print(f"Using column '{ppg_column}' as PPG signal (best guess)")

        # Ensure PPG data is numeric and clean
        data[ppg_column] = pd.to_numeric(data[ppg_column], errors='coerce')
        if data[ppg_column].isna().any():
            data[ppg_column].interpolate(method='linear', inplace=True)

        # --- 2. Process PPG signal using neurokit2 ---
        try:
            signals, info = nk.ppg_process(data[ppg_column], sampling_rate=SAMPLING_RATE)

            required_columns = ['PPG_Raw', 'PPG_Clean', 'PPG_Rate', 'PPG_Quality', 'PPG_Peaks']
            for col in required_columns:
                if col not in signals.columns:
                    if col == 'PPG_Raw':
                        signals['PPG_Raw'] = data[ppg_column].values
                    elif col == 'PPG_Quality':
                        signals['PPG_Quality'] = nk.signal_quality(signals['PPG_Clean'], method="zhao2018")
                
            signals['DateTime'] = data['DateTime'].values

            peak_indices = np.where(signals['PPG_Peaks'] == 1)[0]
            print(f"Found {len(peak_indices)} peaks in {file_path.name}")

            if len(peak_indices) <= 10:
                print(f"Insufficient peaks ({len(peak_indices)}) for reliable HRV calculation in {file_path.name}")
                return
                
        except Exception as e:
            print(f"Error during initial PPG processing: {e}")
            traceback.print_exc()
            return

        print(f"\n=== RAW TIMESTAMPS FROM FILE ===")
        for i in range(min(5, len(data))):
            print(f"Index {i}: {data['DateTime'].iloc[i]}")
    
        # --- 3. Use sliding window processing ---
        total_samples = len(data)

        print(f"\n=== TIMESTAMP VERIFICATION ===")
        print(f"First record timestamp: {data['DateTime'].iloc[0]}")
        first_timestamp = data['DateTime'].iloc[0]
        print(f"Using first timestamp: {first_timestamp}")

        first_window_start_idx = 0

        window_shift_samples = WINDOW_SHIFT * SAMPLING_RATE

        num_windows = max(1, (total_samples - WINDOW_SAMPLES) // window_shift_samples + 1)
        print(f"Processing {num_windows} sliding windows")

        all_window_results = []

        # --- 4. Process each window ---
        for window_idx in range(num_windows):
            try:
                start_idx = first_window_start_idx + (window_idx * window_shift_samples)
                end_idx = min(start_idx + WINDOW_SAMPLES, total_samples)

                if end_idx > total_samples:
                    end_idx = total_samples
                    if (end_idx - start_idx) < (WINDOW_SAMPLES * 0.6):
                        break

                window_time = data['DateTime'].iloc[start_idx]

                if window_idx < 3:
                    print(f"Window {window_idx}: start_idx={start_idx}, time={window_time}")

                if hasattr(window_time, 'replace'):
                    window_time = window_time.replace(microsecond=0)

                window_peak_indices = peak_indices[(peak_indices >= start_idx) & (peak_indices < end_idx)]

                window_peak_indices_adjusted = window_peak_indices - start_idx

                min_peaks = 10
                if window_idx == num_windows - 1:  
                    min_peaks = 5  ำ

                if len(window_peak_indices_adjusted) < min_peaks:
                    continue

                window_result = {
                    'DateTime': window_time,
                    'NumPeaks': len(window_peak_indices_adjusted)
                }

                window_slice = slice(start_idx, end_idx)
                for col in ['PPG_Raw', 'PPG_Clean', 'PPG_Rate', 'PPG_Quality', 'PPG_Peaks']:
                    if col in signals.columns:
                        window_result[col] = signals[col][window_slice].mean()

                hrv_metrics = {}

                try:
                    if len(window_peak_indices_adjusted) >= 30:
                        hrv_time = nk.hrv_time(window_peak_indices_adjusted, sampling_rate=SAMPLING_RATE, show=False)
                        hrv_freq = nk.hrv_frequency(window_peak_indices_adjusted, sampling_rate=SAMPLING_RATE, show=False)
                        hrv_nonlinear = nk.hrv_nonlinear(window_peak_indices_adjusted, sampling_rate=SAMPLING_RATE, show=False)

                        hrv_df = pd.concat([hrv_time, hrv_freq, hrv_nonlinear], axis=1)

                        for col in hrv_df.columns:
                            hrv_metrics[col] = hrv_df[col].values[0] if not pd.isna(hrv_df[col].values[0]) else np.nan
                    else:
                        hrv_time = nk.hrv_time(window_peak_indices_adjusted, sampling_rate=SAMPLING_RATE, show=False)

                        freq_metrics = ['HRV_ULF', 'HRV_VLF', 'HRV_LF', 'HRV_HF', 'HRV_VHF', 'HRV_TP',
                                       'HRV_LFHF', 'HRV_LFn', 'HRV_HFn', 'HRV_LnHF']

                        for col in hrv_time.columns:
                            hrv_metrics[col] = hrv_time[col].values[0] if not pd.isna(hrv_time[col].values[0]) else np.nan

                        for col in freq_metrics:
                            hrv_metrics[col] = np.nan
                except Exception as e:
                    print(f"Warning: Full HRV calculation failed, using minimal calculation: {e}")
                    hrv_metrics = calculate_hrv_metrics(window_peak_indices_adjusted, SAMPLING_RATE, minimal=True)
                
                if not hrv_metrics:
                    continue
                
                for key, value in hrv_metrics.items():
                    window_result[key] = value

                if 'HR' not in window_result and len(window_peak_indices_adjusted) >= 2:
                    rr_intervals = np.diff(window_peak_indices_adjusted) / SAMPLING_RATE
                    window_result['HR'] = 60 / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else np.nan
                    window_result['RR_Mean'] = np.mean(rr_intervals) * 1000  # in ms
                    window_result['RMSSD'] = np.sqrt(np.mean(np.diff(rr_intervals * 1000) ** 2)) if len(rr_intervals) > 1 else np.nan
                    window_result['SDNN'] = np.std(rr_intervals * 1000)  # in ms

                all_window_results.append(window_result)
            except Exception as e:
                print(f"Error processing window {window_idx+1}: {e}")
                continue
            
        # --- 5. Combine all window results ---
        if not all_window_results:
            print(f"No valid windows for {file_path.name}")
            return

        processed_data = pd.DataFrame(all_window_results)

        required_column_list = [
            'DateTime', 'NumPeaks', 'PPG_Raw', 'PPG_Clean', 'PPG_Rate', 'PPG_Quality', 'PPG_Peaks',
            'HR', 'RR_Mean', 'RMSSD', 'SDNN',
        ]
        
        for col in required_column_list:
            if col not in processed_data.columns and col != 'DateTime':
                processed_data[col] = np.nan

        if 'DateTime' in processed_data.columns:
            processed_data.set_index('DateTime', inplace=True)

        for key, value in subject_label.items():
            processed_data[key] = value
            
        # --- 6. Save processed data ---
        try:
            processed_data.to_csv(output_path, index=True)
            print(f'Successfully processed {file_path.name} and saved to {output_path}')
        except Exception as save_error:
            print(f"Error saving to {output_path}: {save_error}")
            # Try alternative location in user home directory
            alt_path = Path(os.path.expanduser("~")) / "PPG_Processed"
            alt_path.mkdir(parents=True, exist_ok=True)
            alt_output = alt_path / output_filename
            processed_data.to_csv(alt_output, index=True)
            print(f'Saved to alternative location: {alt_output}')
        

        ENABLE_PLOTS = True 
        if ENABLE_PLOTS:
            try:
                plt.figure(figsize=(12, 6))
                time_domain_candidates = ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD']
                time_metrics = [m for m in time_domain_candidates if m in processed_data.columns]
                
                if time_metrics:
                    for metric in time_metrics:
                        plt.plot(processed_data.index, processed_data[metric], label=metric.replace('HRV_', ''))
                    plt.title(f'HRV Time Domain Metrics for {subject_id}')
                    plt.xlabel('Time')
                    plt.ylabel('ms')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    
                    time_plot_path = PROCESSED_PATH / f"{file_path.stem}_time_metrics.png"
                    plt.savefig(time_plot_path)
                    print(f"Time domain plot saved to {time_plot_path}")
                    plt.close()
                
                # Frequency domain plot
                plt.figure(figsize=(12, 6))
                freq_domain_candidates = ['HRV_HF', 'HRV_LF', 'HRV_LFHF']
                freq_metrics = [m for m in freq_domain_candidates if m in processed_data.columns]
                
                if freq_metrics:
                    for metric in freq_metrics:
                        plt.plot(processed_data.index, processed_data[metric], label=metric.replace('HRV_', ''))
                    plt.title(f'HRV Frequency Domain Metrics for {subject_id}')
                    plt.xlabel('Time')
                    plt.ylabel('Power (ms²)')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    
                    # Save frequency domain plot
                    freq_plot_path = PROCESSED_PATH / f"{file_path.stem}_freq_metrics.png"
                    plt.savefig(freq_plot_path)
                    print(f"Frequency domain plot saved to {freq_plot_path}")
                    plt.close()
                
                # Heart rate plot
                if 'HR' in processed_data.columns:
                    plt.figure(figsize=(12, 6))
                    plt.plot(processed_data.index, processed_data['HR'], label='Heart Rate', color='red')
                    plt.title(f'Heart Rate for {subject_id}')
                    plt.xlabel('Time')
                    plt.ylabel('BPM')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    
                    # Save HR plot
                    hr_plot_path = PROCESSED_PATH / f"{file_path.stem}_heart_rate.png"
                    plt.savefig(hr_plot_path)
                    print(f"Heart rate plot saved to {hr_plot_path}")
                    plt.close()
                    
            except Exception as plot_error:
                print(f"Error generating plots: {plot_error}")
                traceback.print_exc()
        
        # Print processing time
        processing_time = time.time() - start_time
        print(f"Processing completed in {processing_time:.2f} seconds")
            
    except Exception as e:
        print(f'Error processing {file_path.name}: {str(e)}')
        traceback.print_exc()


def main():
    """Main function to process all PPG files"""
    files = list(RAW_PATH.glob('*.csv'))

    if not files:
        print(f'No CSV files found in {RAW_PATH}')
        return

    print(f'Found {len(files)} files to process')

    for file_path in files:
        process_ppg_file(file_path)


if __name__ == '__main__':

    main()
