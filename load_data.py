import os
import tarfile
import shutil

input_dir = r"C:\Users\pauls\Downloads\frames_output_tar"
output_dir = r"D:\gi-project\dataset"

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

# Loop through all .tar.gz files
for file_name in os.listdir(input_dir):
    if file_name.endswith(".tar.gz"):
        file_path = os.path.join(input_dir, file_name)
        
        # Folder name (remove .tar.gz)
        folder_name = os.path.splitext(os.path.splitext(file_name)[0])[0]
        extract_path = os.path.join(output_dir, folder_name)
        
        # Create folder for this archive
        os.makedirs(extract_path, exist_ok=True)
        
        print(f"Extracting {file_name} -> {extract_path}")
        
        # Extract safely
        try:
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(path=extract_path)
        except Exception as e:
            print(f"Error extracting {file_name}: {e}")

print("✅ Extraction completed for all .tar.gz files!")
