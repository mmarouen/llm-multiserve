import os
from google.cloud import storage
from huggingface_hub import snapshot_download

model_id = "meta-llama/Llama-3.1-8B-Instruct"
token_id = 'REMOVED'
local_model_parent = '/Users/marouenazzouz/Documents/clean_models'
model_name_folder = model_id.replace("/", "-")
local_model_dir = os.path.join(local_model_parent, model_name_folder)
gs_bucket = 'gcp-ml-models'
gs_relative_path = model_name_folder
print(f"Downloading {model_id}...")
snapshot_download(
    repo_id=model_id,
    token=token_id,
    local_dir=os.path.join(local_model_dir, ),
    local_dir_use_symlinks=False,  # This fixes your symlink issue!
    ignore_patterns=["*.msgpack", "*.h5"] # Optional: ignore formats you don't need
)
client = storage.Client()
bucket = client.bucket(gs_bucket)

uploaded_files = 0
for root, _, files in os.walk(local_model_dir):
    for file in files:
        local_path = os.path.join(root, file)
        
        # Maintain folder structure in GCS
        rel_path = os.path.relpath(local_path, local_model_dir)
        gcs_blob_path = os.path.join(gs_relative_path, rel_path)
        
        blob = bucket.blob(gcs_blob_path)
        print(f"Uploading {local_path} to gs://{gs_bucket}/{gcs_blob_path}")
        blob.upload_from_filename(local_path)
        uploaded_files += 1
        
        # Optional: clean up the local file after upload so it isn't 
        # re-uploaded on the next profile stop
        os.remove(local_path)
