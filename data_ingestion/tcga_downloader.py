import os
import requests
import pandas as pd

# Base URL for GDC API
GDC_BASE = "https://api.gdc.cancer.gov/"

def search_tcga_projects(project="TCGA-LUAD", max_files=5):
    """
    Search TCGA project for slide files (default: Lung Adenocarcinoma)
    """
    endpoint = f"{GDC_BASE}files"
    params = {
        "filters": '{"op":"and","content":[{"op":"=","content":{"field":"cases.project.project_id","value":"' + project + '"}},{"op":"=","content":{"field":"data_format","value":"SVS"}}]}',
        "format": "JSON",
        "size": max_files
    }
    r = requests.get(endpoint, params=params)
    r.raise_for_status()
    data = r.json()
    file_records = [
        {
            "file_id": f["file_id"],
            "file_name": f["file_name"],
            "case_id": f["cases"][0]["case_id"]
        }
        for f in data["data"]["hits"]
    ]
    return pd.DataFrame(file_records)

def download_tcga_file(file_id, out_dir="tcga_data"):
    """
    Download a TCGA slide by file_id
    """
    os.makedirs(out_dir, exist_ok=True)
    url = f"{GDC_BASE}data/{file_id}"
    response = requests.get(url, stream=True)
    response.raise_for_status()
    out_path = os.path.join(out_dir, f"{file_id}.svs")
    with open(out_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return out_path

if __name__ == "__main__":
    df = search_tcga_projects(project="TCGA-LUAD", max_files=2)
    print(df)
    for fid in df["file_id"]:
        path = download_tcga_file(fid)
        print("Downloaded:", path)
