import pydantic
from pydantic import BaseModel

class UserMetrics(BaseModel):
    output_tokens: int=None
    input_tokens: int=None
    ttft: float=None
    latency: float=None
    tps: float=None
    def to_dict(self):
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()