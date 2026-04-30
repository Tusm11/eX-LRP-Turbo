import sys
import os
import yaml
import json
import torch
import torch.nn.functional as F
import functools
import time
import gc
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import override
from safetensors import safe_open

# --- WINDOWS RESILIENCE PATCH ---
_original_rename = os.rename
def _resilient_rename(src, dst):
    for i in range(10):
        try:
            if os.path.exists(dst): 
                try: os.remove(dst)
                except Exception: pass
            return _original_rename(src, dst)
        except PermissionError:
            if i == 9: raise
            time.sleep(2)
os.rename = _resilient_rename

# --- MERGEKIT 0.1.4 IMPORTS ---
from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import ConfigParameterDef, MergeMethod, MergeTensorInput
from mergekit.merge_methods.registry import REGISTERED_MERGE_METHODS
from mergekit.options import MergeOptions
from mergekit.config import MergeConfiguration
from mergekit.merge import run_merge

# Windows Concurrency Shield
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ---------------- THE ULTIMATE PYDANTIC FIX: GLOBAL INJECTION ----------------
import mergekit
for mod_name, module in list(sys.modules.items()):
    if mod_name.startswith("mergekit"):
        try: setattr(module, "torch", torch)
        except Exception: pass

try:
    from mergekit.architecture.base import ConfiguredModelArchitecture, ConfiguredModuleArchitecture
    ConfiguredModuleArchitecture.model_rebuild()
    ConfiguredModelArchitecture.model_rebuild()
except Exception: pass

# ---------------- GLOBAL CACHED ROTATION MATRIX ----------------
@functools.lru_cache(maxsize=1)
def get_cached_q_t(block_size: int, device: str):
    torch.manual_seed(42)
    R = torch.randn(block_size, block_size, device=device)
    Q, _ = torch.linalg.qr(R)
    return Q.t().contiguous()

# ---------------- TURBO SURGICAL LOADER ----------------
def get_tensor_safely(path: str, tensor_name: str) -> Optional[torch.Tensor]:
    if not os.path.exists(path): return None
    # RETRY LOGIC for Windows File Locking (Anti-Virus)
    for i in range(5):
        try:
            with safe_open(path, framework="pt", device="cpu") as f:
                if tensor_name in f.keys():
                    return f.get_tensor(tensor_name).clone()
            return None
        except Exception:
            if i == 4: return None
            time.sleep(0.5)
    return None

@functools.lru_cache(maxsize=1)
def load_manifest():
    manifest_path = "turbo_work/turbo_manifest.json"
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            return json.load(f)
    return {}

# ---------------- TURBO DEQUANT ----------------
def dequantize_turbo_vault(layer_name: str, turbo_path: str, device: str = "cpu") -> torch.Tensor:
    hero_weights = get_tensor_safely(turbo_path, f"{layer_name}.hero_weights")
    if hero_weights is None: return None
    
    hero_mask = get_tensor_safely(turbo_path, f"{layer_name}.hero_mask")
    shape_tensor = get_tensor_safely(turbo_path, f"{layer_name}.shape")
    if hero_mask is None or shape_tensor is None: return None
    
    shape = shape_tensor.tolist()
    full_tensor = torch.zeros(tuple(shape), device=device)
    
    noise_quant = get_tensor_safely(turbo_path, f"{layer_name}.noise_quant")
    if noise_quant is not None:
        noise_scales = get_tensor_safely(turbo_path, f"{layer_name}.noise_scales")
        block_size = 1024
        orig_len = noise_quant.numel()
        padding = (block_size - (orig_len % block_size)) % block_size
        noise_padded = F.pad(noise_quant.float(), (0, padding))
        num_blocks = noise_padded.numel() // block_size
        Q_t = get_cached_q_t(block_size, device)
        scale = noise_scales[0].item() if noise_scales is not None else 1.0
        dequant_blocks = torch.matmul(noise_padded.view(num_blocks, block_size) * scale, Q_t)
        full_tensor[~hero_mask] = dequant_blocks.flatten()[:orig_len]
        del dequant_blocks, noise_padded
        
    full_tensor[hero_mask] = hero_weights.float()
    del hero_weights, hero_mask
    return full_tensor

# ---------------- GLOBAL COUNTER ----------------
_TASK_COUNTER = 0

# ---------------- MERGE TASK ----------------
class LRPTurboMergeTask(Task[torch.Tensor], frozen=True):
    gather_tensors: MergeTensorInput
    base_model: Optional[ModelReference]
    model_weights: ImmutableMap[ModelReference, float]
    density: float
    weight_info: WeightInfo

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> torch.Tensor:
        global _TASK_COUNTER
        _TASK_COUNTER += 1
        
        # IRON-MAN STABILIZATION BREAK
        # Every 100 tasks, we pause for 2 seconds to let the SSD catch up
        if _TASK_COUNTER % 100 == 0:
            gc.collect()
            time.sleep(2.0)
        else:
            time.sleep(0.01)
        
        base_raw = tensors.get(self.base_model) if self.base_model else None
        if base_raw is None:
            first_tensor = list(tensors.values())[0] if tensors else None
            base_tensor = torch.zeros_like(first_tensor) if first_tensor is not None else None
        else:
            base_tensor = base_raw.clone()

        manifest = load_manifest()
        merged_deltas = None
        
        name = self.weight_info.name.lower()
        is_critical = any(x in name for x in ["norm", "embed", "ln_", "head", "bias"])
        current_density = 1.0 if is_critical else self.density

        for ref, tensor in tensors.items():
            if ref == self.base_model: continue
            
            ref_str = str(ref)
            ref_data = manifest.get(ref_str, {})
            turbo_path, lrp_path = ref_data.get("turbo_model"), ref_data.get("lrp_scores")

            weight = None
            if turbo_path:
                weight = dequantize_turbo_vault(self.weight_info.name, turbo_path, device=tensor.device.type)
            
            if weight is None: weight = tensor.clone()
            
            delta = weight.sub_(base_tensor)
            
            importance = None
            if lrp_path and not is_critical:
                importance = get_tensor_safely(lrp_path, self.weight_info.name)

            if importance is None or is_critical: 
                importance = delta.abs()

            k = int(current_density * importance.numel())
            if k >= importance.numel():
                mask = torch.ones_like(importance)
            else:
                mask = torch.zeros_like(importance)
                if k > 0:
                    _, topk = torch.topk(importance.view(-1), k)
                    mask.view(-1)[topk] = 1
            
            del importance
            w_val = self.model_weights[ref] if ref in self.model_weights else 1.0
            delta.mul_(mask).mul_(w_val)
            del mask

            if merged_deltas is None: merged_deltas = delta
            else: 
                merged_deltas.add_(delta)
                del delta
            
            gc.collect()

        if merged_deltas is None: merged_deltas = torch.zeros_like(base_tensor)
        res = base_tensor.add_(merged_deltas)
        del merged_deltas, base_tensor
        return res

# ---------------- MERGE METHOD ----------------
class LRPTurboMerge(MergeMethod):
    def name(self) -> str: return "lrp_turbo"
    def parameters(self) -> List[ConfigParameterDef]:
        return [ConfigParameterDef(name="density", required=False, default_value=0.7)]
    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [ConfigParameterDef(name="weight", required=False, default_value=1.0)]

    @override
    def make_task(self, **kwargs) -> Task:
        model_weights_dict = {}
        t_params = kwargs["tensor_parameters"]
        for model_ref, params in t_params.items():
            if model_ref != kwargs["base_model"]:
                model_weights_dict[model_ref] = params["weight"] if "weight" in params else 1.0

        m_params = kwargs["parameters"]
        density_val = m_params["density"] if "density" in m_params else 0.5

        return LRPTurboMergeTask(
            gather_tensors=kwargs["tensors"],
            base_model=kwargs["base_model"],
            model_weights=ImmutableMap(model_weights_dict),
            density=density_val,
            weight_info=kwargs["output_weight"],
        )

# ---------------- MAIN ----------------
def main():
    if len(sys.argv) < 3: return
    config_path, output_path = sys.argv[1], sys.argv[2]
    REGISTERED_MERGE_METHODS["lrp_turbo"] = LRPTurboMerge()
    
    options = MergeOptions(
        device="cpu", 
        low_cpu_memory=True, 
        trust_remote_code=True, 
        copy_tokenizer=True,
        num_workers=1,
        lazy_unpickle=True 
    )
    
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
    if "models" in raw_config:
        for m in raw_config["models"]:
            if "parameters" in m:
                m["parameters"].pop("lrp_scores", None); m["parameters"].pop("turbo_model", None)
    
    merge_config = MergeConfiguration.model_validate(raw_config)
    
    try:
        import psutil
        p = psutil.Process(os.getpid())
        p.nice(psutil.HIGH_PRIORITY_CLASS)
    except Exception: pass
    
    print(f"Starting Iron-Man Merge into {output_path}...")
    run_merge(merge_config, output_path, options=options)

if __name__ == "__main__":
    main()