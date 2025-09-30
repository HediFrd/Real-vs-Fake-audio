import os
import shutil
from pathlib import Path
from opensoundscape import Audio, Spectrogram

def copy_folder_structure(src_root, dst_root):
    """Copy folder structure without copying files"""
    for root, dirs, _ in os.walk(src_root):
        for dir_name in dirs:
            src_dir = os.path.join(root, dir_name)
            # Create relative path from src_root
            rel_path = os.path.relpath(src_dir, src_root)
            dst_dir = os.path.join(dst_root, rel_path)
            os.makedirs(dst_dir, exist_ok=True)
            print(f"Created directory: {dst_dir}")

def convert_audio_to_spectrograms(src_root, dst_root, shape=(128, 128)):
    """Convert all audio files to spectrograms maintaining folder structure"""
    audio_extensions = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    total_processed = 0
    total_failed = 0
    
    for root, _, files in os.walk(src_root):
        audio_files = [f for f in files if f.lower().endswith(audio_extensions)]
        if not audio_files:
            continue
            
        # Create relative path from src_root
        rel_path = os.path.relpath(root, src_root)
        dst_dir = os.path.join(dst_root, rel_path)
        os.makedirs(dst_dir, exist_ok=True)
        
        print(f"\nProcessing files in {rel_path}...")
        total_files = len(audio_files)
        
        for idx, audio_file in enumerate(audio_files, 1):
            src_file = os.path.join(root, audio_file)
            dst_file = os.path.join(dst_dir, f"{os.path.splitext(audio_file)[0]}.png")
            
            try:
                # Convert to spectrogram
                aud = Audio.from_file(src_file)
                spec = Spectrogram.from_audio(aud)
                img = spec.to_image(shape=shape, invert=False)
                img.save(dst_file)
                
                total_processed += 1
                print(f"[{idx}/{total_files}] Processed: {os.path.relpath(src_file, src_root)} -> {os.path.relpath(dst_file, dst_root)}")
            except Exception as e:
                total_failed += 1
                print(f"Failed to process {audio_file}: {e}")
    
    print(f"\nConversion complete!")
    print(f"Successfully processed: {total_processed} files")
    if total_failed > 0:
        print(f"Failed to process: {total_failed} files")

def main():
    # Define source and destination paths
    dataset_root = Path(r"C:\Egyetem\0_NTU\Hackathon\deep-detect\dataset")
    spectrograms_root = dataset_root.parent / "dataset_spectrograms"
    
    print("Creating folder structure...")
    copy_folder_structure(dataset_root, spectrograms_root)
    
    print("\nConverting audio files to spectrograms...")
    convert_audio_to_spectrograms(dataset_root, spectrograms_root)

if __name__ == "__main__":
    main()