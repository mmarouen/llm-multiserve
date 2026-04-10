from google.cloud import aiplatform

# 1. Create a Dedicated Endpoint
endpoint = aiplatform.Endpoint.create(
    display_name="Triton-llama3.2-3b-trt-llm",
    dedicated_endpoint_enabled=True,
    location="europe-west2"
)