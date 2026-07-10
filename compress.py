import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

input_dir = "data"
output_dir = "data_compressed"
os.makedirs(output_dir, exist_ok=True)

def process_image(filename):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)
    
    try:
        with Image.open(input_path) as img:
            img = img.convert('RGB')
            # Resize image maintaining aspect ratio
            img.thumbnail((300, 300), Image.Resampling.LANCZOS)
            # Save compressed
            img.save(output_path, "JPEG", optimize=True, quality=65)
    except Exception as e:
        print(f"Failed to process {filename}: {e}")

files = os.listdir(input_dir)
print(f"Compressing {len(files)} images...")
with ThreadPoolExecutor(max_workers=8) as executor:
    executor.map(process_image, files)
print("Done!")
