from transformers import LlamaConfig
try:
    import pynvml
except ImportError:
    pynvml = None
    print("pynvml not installed. Some details (NVLink) will be missing.")
import torch

def get_standard_kv_cache(model_config: LlamaConfig, seq_length: int, precision:int=2):
    return 2 * seq_length * model_config.num_hidden_layers * model_config.hidden_size *  precision / 1e9

def describe_gpus():
    
    num_gpus = torch.cuda.device_count()
    description = f"Number of GPUs: {num_gpus}\n"
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        description += f"GPU {i}: {props.name}\n"
        description += f"  Total Memory: {props.total_memory / 1e9:.2f} GB\n"
        description += f"  Compute Capability: {props.major}.{props.minor}\n"
        description += f"  Multi-Processor Count: {props.multi_processor_count}\n"
        description += f"  CUDA Device Index: {i}\n"
    '''
    if pynvml:
        try:
            pynvml.nvmlInit()
            labels = {
                pynvml.NVML_TOPOLOGY_INTERNAL:   "NVLink (same board)",
                pynvml.NVML_TOPOLOGY_SINGLE:     "PIX - same PCIe switch",
                pynvml.NVML_TOPOLOGY_MULTIPLE:   "PXB - multiple PCIe switches",
                pynvml.NVML_TOPOLOGY_HOSTBRIDGE: "PHB - PCIe host bridge",
                pynvml.NVML_TOPOLOGY_CPU:        "CPU - same NUMA/CPU",
                pynvml.NVML_TOPOLOGY_SYSTEM:     "SYS - cross-socket",
            }
            description += "NVML topology:\n"
            for i in range(num_gpus):
                for j in range(i + 1, num_gpus):
                    handle_i = pynvml.nvmlDeviceGetHandleByIndex(i)
                    handle_j = pynvml.nvmlDeviceGetHandleByIndex(j)
                    level = pynvml.nvmlDeviceGetTopologyCommonAncestor(handle_i, handle_j)
                    print(f"GPU{i} <-> GPU{j}: {labels.get(level, level)}")
                    description += f"GPU{i} <-> GPU{j}: {labels.get(level, level)}\n"
            pynvml.nvmlShutdown()
        except pynvml.NVMLError as e:
            description += f"NVML unavailable {e}"
    '''
    return description