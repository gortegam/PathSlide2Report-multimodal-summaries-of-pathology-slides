import os
import openslide
import pandas as pd
from PIL import Image

def tile_wsi(wsi_path, out_dir="tcga_patches", patch_size=512, max_patches=20):
    """
    Extract tiles from a TCGA whole-slide image (.svs).
    
    Args:
        wsi_path (str): Path to the .svs file
        out_dir (str): Directory where patches will be saved
        patch_size (int): Size of each square patch in pixels
        max_patches (int): Limit how many patches to extract (for demo purposes)
    
    Returns:
        metadata (list of dicts): Info about each extracted patch
    """
    os.makedirs(out_dir, exist_ok=True)

    slide = openslide.OpenSlide(wsi_path)
    wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]

    width, height = slide.dimensions
    print(f"Slide: {wsi_name}, size = {width}x{height}")

    metadata = []
    count = 0

    for x in range(0, width, patch_size * 50):   # stride of 50 patches, not dense
        for y in range(0, height, patch_size * 50):
            if count >= max_patches:
                break

            patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
            patch_filename = f"{wsi_name}_x{x}_y{y}.png"
            patch_path = os.path.join(out_dir, patch_filename)
            patch.save(patch_path)

            metadata.append({
                "slide_id": wsi_name,
                "patch_file": patch_filename,
                "x": x,
                "y": y,
                "patch_size": patch_size,
                "level": 0
            })

            count += 1

    return metadata

if __name__ == "__main__":
    # Example usage
    wsi_path = "tcga_data/example_slide.svs"  # replace with a real downloaded file
    patches_dir = "data/patches"
    metadata_csv = "data/patches_metadata.csv"

    metadata_records = tile_wsi(wsi_path, out_dir=patches_dir, patch_size=512, max_patches=10)
    df = pd.DataFrame(metadata_records)
    df.to_csv(metadata_csv, index=False)

    print(f"Saved {len(df)} patches to {patches_dir}")
    print(f"Metadata CSV: {metadata_csv}")
