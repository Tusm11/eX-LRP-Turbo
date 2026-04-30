# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F
from typing_extensions import override
from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
import functools

def build_mask(importance: torch.Tensor, density: float) -> torch.Tensor:
    if density >= 1.0: return torch.ones_like(importance)
    k = int(density * importance.numel())
    if k <= 0: return torch.zeros_like(importance)
    w = importance.view(-1)
    topk = torch.argsort(w, descending=True)[:k]
    mask = torch.zeros_like(importance)
    mask.view(-1)[topk] = 1
    return mask

@functools.lru_cache(maxsize=1)
def get_rotation_matrix(size: int, device: str):
    torch.manual_seed(42)
    R = torch.randn(size, size, device=device)
    Q, _ = torch.linalg.qr(R)
    return Q

def dequantize_turbo(compressed_data: Dict[str, Any], device: str = "cpu") -> torch.Tensor:
    hero_weights = compressed_data["hero_weights"].to(device)
    hero_mask = compressed_data["hero_mask"].to(device)
    noise_quant = compressed_data.get("noise_quant")
    noise_scales = compressed_data.get("noise_scales")
    shape = compressed_data["shape"]
    
    full_tensor = torch.zeros(shape, device=device)
    
    if noise_quant is not None and noise_quant.numel() > 0:
        noise_quant = noise_quant.to(device)
        noise_scales = noise_scales.to(device)
        block_size = 1024
        orig_len = noise_quant.numel()
        padding = (block_size - (orig_len % block_size)) % block_size
        
        noise_padded = F.pad(noise_quant.float(), (0, padding))
        num_blocks = noise_padded.numel() // block_size
        noise_blocks = noise_padded.view(num_blocks, block_size)
        
        Q = get_rotation_matrix(block_size, device)
        scale = noise_scales[0].item()
        dequant_blocks = torch.matmul(noise_blocks * scale, Q.t())
        
        noise_weights = dequant_blocks.flatten()[:orig_len]
        full_tensor[~hero_mask] = noise_weights
        
    full_tensor[hero_mask] = hero_weights.float()
    return full_tensor

@functools.lru_cache(maxsize=8)
def _load_any_tensor_file(path: str):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location="cpu")

class LRPTurboMergeTask(Task[torch.Tensor]):
    gather_tensors: MergeTensorInput
    base_model: Optional[ModelReference]
    model_weights: ImmutableMap[ModelReference, float]
    density: float
    weight_info: WeightInfo
    lrp_scores: Optional[ImmutableMap[str, str]] = None
    turbo_models: Optional[ImmutableMap[str, str]] = None

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> torch.Tensor:
        base_tensor = tensors.get(self.base_model) if self.base_model else None
        if base_tensor is None:
            first_tensor = list(tensors.values())[0] if tensors else None
            base_tensor = torch.zeros_like(first_tensor) if first_tensor is not None else None

        merged_deltas = None
        for ref, tensor in tensors.items():
            if ref == self.base_model: continue
                
            ref_str = str(ref)
            if self.turbo_models and ref_str in self.turbo_models:
                turbo_data = _load_any_tensor_file(self.turbo_models[ref_str])
                if self.weight_info.name in turbo_data:
                    fine_tuned_weight = dequantize_turbo(turbo_data[self.weight_info.name], device=tensor.device)
                else:
                    fine_tuned_weight = tensor
            else:
                fine_tuned_weight = tensor

            delta = fine_tuned_weight - base_tensor
            
            importance = None
            if self.lrp_scores and ref_str in self.lrp_scores:
                scores_dict = _load_any_tensor_file(self.lrp_scores[ref_str])
                importance = scores_dict.get(self.weight_info.name)
                if importance is not None:
                    importance = importance.to(delta.device)

            if importance is None: continue

            mask = build_mask(importance, self.density)
            weight = self.model_weights[ref] if ref in self.model_weights else 1.0
            
            if merged_deltas is None:
                merged_deltas = weight * (delta * mask)
            else:
                merged_deltas += weight * (delta * mask)

        return base_tensor + (merged_deltas if merged_deltas is not None else torch.zeros_like(base_tensor))

class LRPTurboMerge(MergeMethod):
    def name(self) -> str: return "lrp_turbo"
    def parameters(self) -> List[ConfigParameterDef]:
        return [ConfigParameterDef(name="density", required=False, default_value=0.7)]
    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="weight", required=False, default_value=1.0),
            ConfigParameterDef(name="lrp_scores", required=False),
            ConfigParameterDef(name="turbo_model", required=False),
        ]

    @override
    def make_task(self, **kwargs) -> Task:
        output_weight = kwargs["output_weight"]
        tensors = kwargs["tensors"]
        parameters = kwargs["parameters"]
        tensor_parameters = kwargs["tensor_parameters"]
        base_model = kwargs["base_model"]

        model_weights = {}
        lrp_scores_map = {}
        turbo_models_map = {}
        
        for model_ref, params in tensor_parameters.items():
            if model_ref != base_model:
                model_weights[model_ref] = params.get("weight", 1.0)
                if "lrp_scores" in params: lrp_scores_map[str(model_ref)] = str(params["lrp_scores"])
                if "turbo_model" in params: turbo_models_map[str(model_ref)] = str(params["turbo_model"])

        return LRPTurboMergeTask(
            gather_tensors=tensors,
            base_model=base_model,
            model_weights=ImmutableMap(model_weights),
            density=parameters.get("density", 0.7),
            weight_info=output_weight,
            lrp_scores=ImmutableMap(lrp_scores_map) if lrp_scores_map else None,
            turbo_models=ImmutableMap(turbo_models_map) if turbo_models_map else None,
        )
