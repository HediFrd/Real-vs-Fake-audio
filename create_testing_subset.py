import os
import shutil
import random

# Source and destination paths
source_base_path = r"C:\Egyetem\0_NTU\Hackathon\deep-detect\dataset\testing"
dest_base_path = source_base_path

# Define source and destination folders
source_folders = ['fake', 'real']
dest_folders = ['fake_subset', 'real_subset']

# Number of files to sample per class
n_samples = 500

# Process each folder
for source_folder, dest_folder in zip(source_folders, dest_folders):
    # Create full paths
    source_path = os.path.join(source_base_path, source_folder)
    dest_path = os.path.join(dest_base_path, dest_folder)
    
    # Create destination folder if it doesn't exist
    os.makedirs(dest_path, exist_ok=True)
    
    # Get list of all wav files in source folder
    wav_files = [f for f in os.listdir(source_path) if f.endswith('.wav')]
    
    # Check if we have enough files
    if len(wav_files) < n_samples:
        print(f"Warning: Only {len(wav_files)} files available in {source_folder} folder")
        selected_files = wav_files  # Take all files if less than n_samples
    else:
        # Randomly select n_samples files
        selected_files = random.sample(wav_files, n_samples)
    
    # Copy and rename selected files
    for idx, file in enumerate(selected_files, 1):
        source_file = os.path.join(source_path, file)
        new_filename = f"{source_folder}_{idx}.wav"  # Rename to fake_1.wav or real_1.wav format
        dest_file = os.path.join(dest_path, new_filename)
        shutil.copy2(source_file, dest_file)
        
    print(f"Copied and renamed {len(selected_files)} files from {source_folder} to {dest_folder}")

print("Testing subset creation complete!")