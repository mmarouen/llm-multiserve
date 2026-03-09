from pydantic import BaseModel
from transformers import LlamaConfig

def get_endpoint_paths(endpoint: dict, number: str, region: str, project_id: str):
    endpoint_path = f"{endpoint['id']}.{region}-{number}.prediction.vertexai.goog" if endpoint['is-dedicated'] else f"{region}-aiplatform.googleapis.com"
    resource_path = f"projects/{project_id}/locations/{region}/endpoints/{endpoint['id']}"
    return resource_path, endpoint_path

def get_standard_kv_cache(model_config: LlamaConfig, seq_length: int, precision:int=2):
    return 2 * seq_length * model_config.num_hidden_layers * model_config.hidden_size *  precision / 1e9

class UserMetrics(BaseModel):
    output_tokens: int=None
    input_tokens: int=None
    ttft: float=None
    latency: float=None
    tps: float=None