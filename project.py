import onnxruntime as ort
import cv2
import numpy as np
import os
from glob import glob
import time

# Configuration
pic_form = ['.jpeg','.jpg','.png','.JPEG','.JPG','.PNG']
providers = ['CPUExecutionProvider']
model_name = 'AnimeGANv3_Hayao_STYLE_36'
in_dir = 'input'
out_dir = 'output'
max_dimension = 1024  # Max dimension for processing

os.makedirs(in_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

session = ort.InferenceSession(f'{model_name}.onnx', providers=providers)

def process_image(img, x8=True):
    # First resize to limit memory usage
    h, w = img.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))
        print(f"Resized image from {w}x{h} to {new_w}x{new_h} to save memory")
    
    h, w = img.shape[:2]
    if x8:
        def to_8s(x):
            return 256 if x < 256 else x - x % 8
        img = cv2.resize(img, (to_8s(w), to_8s(h)))
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    return img

def load_test_data(image_path):
    try:
        img0 = cv2.imread(image_path)
        if img0 is None:
            raise Exception(f"Failed to load image: {image_path}")
            
        # Keep original dimensions for reference
        original_dimensions = img0.shape[:2]
        
        # Process image (with potential resizing)
        img = process_image(img0)
        img = np.expand_dims(img, axis=0)
        
        return img, original_dimensions
    except Exception as e:
        print(f"Error loading image: {e}")
        raise

def convert(img, scale):
    try:
        x = session.get_inputs()[0].name
        fake_img = session.run(None, {x: img})[0]
        
        # Process in chunks to save memory
        images = (np.squeeze(fake_img) + 1.) / 2 * 255
        images = np.clip(images, 0, 255).astype(np.uint8)
        
        # Resize back to original dimensions
        output_image = cv2.resize(images, (scale[1], scale[0]))
        return cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error converting image: {e}")
        raise

def process_directory():
    print(f"Processing images from {in_dir} to {out_dir}")
    in_files = sorted(glob(f'{in_dir}/*'))
    in_files = [x for x in in_files if os.path.splitext(x)[-1].lower() in pic_form]
    
    if not in_files:
        print(f"No images found in {in_dir}. Please add some images and try again.")
        return
    
    for i, img_path in enumerate(in_files):
        print(f"Processing {i+1}/{len(in_files)}: {img_path}")
        start_time = time.time()
        
        try:
            out_name = f"{out_dir}/{os.path.basename(img_path).split('.')[0]}.jpg"
            mat, scale = load_test_data(img_path)
            res = convert(mat, scale)
            cv2.imwrite(out_name, res)
            
            elapsed = time.time() - start_time
            print(f"Saved to {out_name} (took {elapsed:.2f} seconds)")
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
    
    print(f"All done! Processed {len(in_files)} images.")

if __name__ == "__main__":
    process_directory()
    print(f"\nTo use this script:")
    print(f"1. Place your images in the '{in_dir}' folder")
    print(f"2. Run this script")
    print(f"3. Check the '{out_dir}' folder for results")