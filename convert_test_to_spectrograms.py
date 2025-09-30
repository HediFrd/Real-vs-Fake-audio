from opensoundscape import Audio, Spectrogram
from pathlib import Path

def batch_save_spectrograms_from_testing_subset(testing_root: str, output_root: str = "spectrograms_output_test", use_mel: bool = False, shape=(128, 128)):
    """Converts testing subset audio files to spectrograms.
    Files are sorted into fake/real folders based on their filenames.
    
    Parameters:
    - testing_root: path to the testing_subset folder
    - output_root: base output folder to write images into
    - use_mel: if True, use MelSpectrogram, otherwise linear Spectrogram
    - shape: (width, height) for saved image
    """
    testing_root = Path(testing_root)
    output_root = Path(output_root)

    if not testing_root.exists():
        raise FileNotFoundError(f"testing_root does not exist: {testing_root}")

    # Create output directories
    fake_dir = output_root / "fake"
    real_dir = output_root / "real"
    fake_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    # Supported audio formats
    audio_extensions = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

    # Get list of all audio files
    audio_files = [f for f in sorted(testing_root.iterdir()) 
                  if f.is_file() and f.suffix.lower() in audio_extensions]
    
    total_files = len(audio_files)
    total_processed = 0
    total_failed = 0
    fake_count = 0
    real_count = 0

    print(f"\nFound {total_files} audio files to process...")

    for idx, audio_path in enumerate(audio_files, 1):
        try:
            # Determine if fake or real based on filename
            is_fake = audio_path.stem.startswith('fake')
            out_dir = fake_dir if is_fake else real_dir
            
            # Convert to spectrogram
            aud = Audio.from_file(str(audio_path))
            spec = Spectrogram.from_audio(aud)
            img = spec.to_image(shape=shape, invert=False)
            
            # Save spectrogram
            out_path = out_dir / (audio_path.stem + ".png")
            img.save(out_path)
            
            if is_fake:
                fake_count += 1
            else:
                real_count += 1
                
            total_processed += 1
            print(f"[{idx}/{total_files}] Processed {audio_path.name} -> {out_path}")
            
        except Exception as e:
            print(f"Failed to process {audio_path}: {e}")
            total_failed += 1

    print(f"\nConversion complete!")
    print(f"Successfully processed: {total_processed} files")
    print(f"- Fake spectrograms: {fake_count}")
    print(f"- Real spectrograms: {real_count}")
    if total_failed > 0:
        print(f"Failed to process: {total_failed} files")

if __name__ == "__main__":
    # Process the testing_subset using absolute path
    testing_subset = Path(r"C:\Users\mate\Documents\GitHub\Real-vs-Fake-audio\testing_subset")
    batch_save_spectrograms_from_testing_subset(
        str(testing_subset),
        output_root=str(Path(r"C:\Users\mate\Documents\GitHub\Real-vs-Fake-audio\spectrograms_output_test")),
        use_mel=False
    )
    batch_save_spectrograms_from_testing_subset(
        str(testing_subset),
        output_root=str(repo_root / "spectrograms_output_test"),
        use_mel=False
    )