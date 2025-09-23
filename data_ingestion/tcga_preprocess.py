import os
import openslide
import pandas as pd
from PIL import Image

def extract_slide_metadata(slide):
    """
    Extract metadata from OpenSlide properties (varies by scanner).
    """
    props = slide.properties
    metadata = {
        "tissue": props.get("tissue", "unknown"),
        "stain": props.get("aperio.AppMag", "H&E"),  # fallback: most TCGA are H&E
        "magnification": props.get("aperio.AppMag", "unknown"),
        "scanner": props.get("aperio.ScanScope ID", "unknown")
    }
    return metadata

def tile_wsi(wsi_path, out_dir="data/patches", patch_size=512, max_patches=20):
    """
    Extract tiles from a TCGA WSI (.svs) and enrich with metadata.
    """
    os.makedirs(out_dir, exist_ok=True)

    slide = openslide.OpenSlide(wsi_path)
    wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]

    width, height = slide.dimensions
    print(f"Slide: {wsi_name}, size = {width}x{height}")

    # Extract metadata
    slide_metadata = extract_slide_metadata(slide)

    metadata_records = []
    count = 0

    for x in range(0, width, patch_size * 50):   # sparse sampling
        for y in range(0, height, patch_size * 50):
            if count >= max_patches:
                break

            patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
            patch_filename = f"{wsi_name}_x{x}_y{y}.png"
            patch_path = os.path.join(out_dir, patch_filename)
            patch.save(patch_path)

            metadata_records.append({
                "slide_id": wsi_name,
                "patch_file": patch_filename,
                "x": x,
                "y": y,
                "patch_size": patch_size,
                "level": 0,
                "tissue": slide_metadata.get("tissue", "unknown"),
                "stain": slide_metadata.get("stain", "H&E"),
                "magnification": slide_metadata.get("magnification", "unknown"),
                "scanner": slide_metadata.get("scanner", "unknown")
            })

            count += 1

    return metadata_records

if __name__ == "__main__":
    wsi_path = "tcga_data/example_slide.svs"  # replace with your downloaded slide
    patches_dir = "data/patches"
    metadata_csv = "data/patches_metadata.csv"

    metadata_records = tile_wsi(wsi_path, out_dir=patches_dir, patch_size=512, max_patches=10)
    df = pd.DataFrame(metadata_records)
    df.to_csv(metadata_csv, index=False)

    print(f"Saved {len(df)} patches to {patches_dir}")
    print(f"Metadata CSV: {metadata_csv}")

