import pandas as pd
import numpy as np
import neurokit2 as nk
import glob
from pathlib import Path
from datetime import datetime
import json

def load_json(file_path: Path) -> dict:
    """Load JSON file"""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data['label']

# Constants
SAMPLING_RATE = 15
RAW_PATH = Path('Raw/eda')
PROCESSED_PATH = Path('Processed/eda')
Label = load_json("label.json")


def process_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df['DateTime'] = pd.to_datetime(df['LocalTimestamp'] , unit='s').dt.tz_localize('UTC').dt.tz_convert('Asia/Bangkok')
    return df

def get_subject_id_from_filename(filename: str) -> str:
    """Extract subject ID from filename (e.g., 'S01_PPG.csv' -> 'S01')"""
    return filename.split('_')[0].upper()

def get_label_for_subject(subject_id: str, labels: list) -> dict:
    """Get label information for a specific subject"""
    subject_id = subject_id.upper()
    for label in labels:
        if label['id'].upper() == subject_id:
            return label
    return None

# Resampling ข้อมูลเป็นทุก 1 วินาที
def process_eda_file(file_path: Path) -> None:
    """Process a single PPG file with label information"""
    try:
        subject_id = get_subject_id_from_filename(file_path.name)
        subject_label = get_label_for_subject(subject_id, Label)
        
        if not subject_label:
            print(f"Warning: No label found for subject {subject_id}")
            return
            
        data = pd.read_csv(file_path)
        data = data.dropna().reset_index(drop=True)
        
        data = process_timestamp(data)
        
        signals, info = nk.eda_process(data['EA'], sampling_rate=SAMPLING_RATE)
        signals = pd.DataFrame(signals)
        
        signals['DateTime'] = data['DateTime']
        
        if len(signals) != len(data):
            print(f"Warning: Length mismatch in {file_path.name}")
            return
        
        signals.set_index('DateTime', inplace=True)
        
        object_cols = {}
        for col in signals.columns:
            if signals[col].dtype == 'object':
                object_cols[col] = signals[col].iloc[0]
        
        resampled_data = signals.resample('1s').mean()
        resampled_data = resampled_data.ffill()
        
        for key, value in subject_label.items():
            resampled_data[key] = value

        for col, value in object_cols.items():
            resampled_data[col] = value

        resampled_data.reset_index(inplace=True)

        PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

        output_path = PROCESSED_PATH / file_path.name
        resampled_data.to_csv(output_path, index=False)
        print(f'Successfully processed {file_path.name} for subject {subject_id}')
        
    except Exception as e:
        import traceback
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
        process_eda_file(file_path)

if __name__ == '__main__':

    main()
