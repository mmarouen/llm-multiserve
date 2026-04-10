from transformers import AutoTokenizer, AutoModelForCausalLM
from .health_check import HealthCheck

tokenizer: AutoTokenizer = None
model: AutoModelForCausalLM = None
engine = None
use_vllm: bool = False
use_trtllm: bool = False
health_checker = HealthCheck()