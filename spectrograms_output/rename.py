import os

BASE_DIR = os.getcwd()  # current folder where script is run
REAL_DIR = os.path.join(BASE_DIR, "real")
FAKE_DIR = os.path.join(BASE_DIR, "fake")

def rename_files(folder, prefix):
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    for idx, filename in enumerate(sorted(files), start=1):
        ext = os.path.splitext(filename)[1]  # keep original extension
        new_name = f"{prefix}_{idx}{ext}"
        src = os.path.join(folder, filename)
        dst = os.path.join(folder, new_name)
        os.rename(src, dst)
        print(f"Renamed: {filename} -> {new_name}")

rename_files(REAL_DIR, "real")
rename_files(FAKE_DIR, "fake")

print("All files renamed successfully.")
