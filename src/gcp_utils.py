import os
from google.cloud import storage
from google.cloud import aiplatform

def get_gcp_endpoint_paths(endpoint: dict, project: dict, region: str):
    number = project['number']
    project_id = project['id']
    #region = project['region']
    endpoint_id = endpoint['id'][region]
    endpoint_path = f"{endpoint_id}.{region}-{number}.prediction.vertexai.goog" if endpoint['is-dedicated'] else f"{region}-aiplatform.googleapis.com"
    resource_path = f"projects/{project_id}/locations/{region}/endpoints/{endpoint_id}"
    return f"https://{endpoint_path}/v1/{resource_path}/invoke" if endpoint['is-dedicated'] else f"https://{endpoint_path}/v1/{resource_path}"

def get_latest_model_version(project, region, model_id):
    aiplatform.init(project=project, location=region)
    model_registry = aiplatform.Model(model_name=model_id)
    versions = model_registry.versioning_registry.list_versions()
    ids = [int(v.version_id) for v in versions]
    return max(ids)

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