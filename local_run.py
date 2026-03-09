import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

model_id = "meta-llama/Llama-3.2-3B-Instruct"
token_id = ''
model_output_dir = '/Users/marouenazzouz/Documents/clean_models'
'''
device = torch.device('mps')
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=token_id,
    dtype=torch.float16
    ).to(device)
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=token_id
    )

input_txt = "Whats the capital of paris?"
inputs = tokenizer(input_txt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=50)
result = tokenizer.decode(outputs[0], skip_special_tokens = True)
print(result)
'''
print(f"Downloading {model_id}...")
snapshot_download(
    repo_id=model_id,
    token=token_id,
    local_dir=os.path.join(model_output_dir, model_id.replace("/", "-")),
    local_dir_use_symlinks=False,  # This fixes your symlink issue!
    ignore_patterns=["*.msgpack", "*.h5"] # Optional: ignore formats you don't need
)
