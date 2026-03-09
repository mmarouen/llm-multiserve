from google.cloud import aiplatform

# 1. Create a Dedicated Endpoint
endpoint = aiplatform.Endpoint.create(
    display_name="Inf-endpt-llama3.2-3b-vllm",
    dedicated_endpoint_enabled=True,
    location="europe-west3"
)