import requests
import pandas as pd

GDC_BASE = "https://api.gdc.cancer.gov/"

def fetch_slide_metadata(project="TCGA-LUAD", max_cases=5):
    """
    Fetch slide-level metadata from the TCGA GDC API.
    """
    endpoint = f"{GDC_BASE}cases"
    params = {
        "filters": f'{{"op":"=","content":{{"field":"project.project_id","value":"{project}"}}}}',
        "expand": "diagnoses,samples,slides",
        "size": max_cases
    }
    r = requests.get(endpoint, params=params)
    r.raise_for_status()
    data = r.json()["data"]["hits"]

    records = []
    for case in data:
        case_id = case["case_id"]
        diagnosis = case.get("diagnoses", [{}])[0].get("diagnosis", "unknown")
        sample_type = case.get("samples", [{}])[0].get("sample_type", "unknown")
        slides = case.get("slides", [])

        for slide in slides:
            records.append({
                "case_id": case_id,
                "slide_id": slide.get("submitter_id", "unknown"),
                "diagnosis": diagnosis,
                "sample_type": sample_type,
                "tissue": case.get("primary_site", "unknown")
            })
    return pd.DataFrame(records)

if __name__ == "__main__":
    df = fetch_slide_metadata(project="TCGA-LUAD", max_cases=10)
    print(df.head())

    # Optionally, join with patch metadata
    try:
        patch_df = pd.read_csv("data/patches_metadata.csv")
        enriched = patch_df.merge(df, on="slide_id", how="left")
        enriched.to_csv("data/patches_metadata_enriched.csv", index=False)
        print("Enriched metadata saved to data/patches_metadata_enriched.csv")
    except FileNotFoundError:
        print("⚠️ No patches_metadata.csv found. Run tcga_preprocess.py first.")
