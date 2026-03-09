import os
import time
import requests
from fastapi import Response
from google.cloud import storage

SYSTEM_PROMPT = 'You are a helpful and concise AI assistant.'

def run_pipeline_rest(payload_dict, endpoint_path, resource_path, session, headers=None, stream=True):

    #url = f"https://{endpoint_path}/v1/{resource_path}:{'streamRawPredict' if stream else 'rawPredict'}"
    url = f"https://{endpoint_path}/v1/{resource_path}/invoke/predict"
    #url = f"https://{endpoint_path}/v1/{resource_path}/predict"
    response = None
    if headers:
        response = requests.post(url, headers=headers, json=payload_dict, stream=stream, timeout=(5.0, 60.0))
    else:
        response = session.post(url, json=payload_dict, stream=stream, timeout=(5.0, 60.0))

    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}: {response.text}")
    if stream:
        # Iterate over the raw HTTP bytes as they arrive
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                yield line
    else:
        yield response.json()

def run_tracing(endpoint_path, resource_path, session, stop=False):
    url = f"https://{endpoint_path}/v1/{resource_path}/invoke/{'stop_profiling' if stop else 'start_profiling'}"
    response = session.post(url, json={}, 
                            #timeout=(5.0, 60.0)
                            )

class HealthCheck:
    def __init__(self):
        self.is_ready = False

    async def __call__(self):
        if not self.is_ready:
            return Response(status_code=503)
        return {"status": "healthy", "model_ready": self.is_ready}

def download_gcs_folder(bucket_name, prefix, dest_dir):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        file_path = os.path.join(dest_dir, os.path.relpath(blob.name, prefix))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        blob.download_to_filename(file_path)

def format_prompt(tokenizer, input_text):
    messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text}
        ]
    return tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

def export_profile_gcp(gs_bucket: str, gs_relative_path: str, profile_folder: str):
    """Recursively uploads all trace files in the local profile folder to GCS."""
    if not os.path.exists(profile_folder):
        print(f"Warning: Profile folder {profile_folder} does not exist. Nothing to export.")
        return

    try:
        # Client automatically picks up Vertex AI endpoint service account credentials
        client = storage.Client()
        bucket = client.bucket(gs_bucket)

        uploaded_files = 0
        for root, _, files in os.walk(profile_folder):
            for file in files:
                local_path = os.path.join(root, file)
                
                # Maintain folder structure in GCS
                rel_path = os.path.relpath(local_path, profile_folder)
                gcs_blob_path = os.path.join(gs_relative_path, rel_path)
                
                blob = bucket.blob(gcs_blob_path)
                print(f"Uploading {local_path} to gs://{gs_bucket}/{gcs_blob_path}")
                blob.upload_from_filename(local_path)
                uploaded_files += 1
                
                # Optional: clean up the local file after upload so it isn't 
                # re-uploaded on the next profile stop
                os.remove(local_path)

        print(f"Export complete. Uploaded {uploaded_files} trace files.")
    except Exception as e:
        print(f"Failed to export profiles to GCS: {e}")