import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List

class LRPComputer:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        print(f"Loading model from {model_path} onto {device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    def compute_all_relevance_scores(self) -> Dict[str, torch.Tensor]:
        """Compute activation-magnitude relevance scores using forward hooks."""
        relevance_scores = {}
        hooks = []
        
        def hook_fn(name):
            def fn(module, input, output):
                with torch.no_grad():
                    # Capture output activation magnitude
                    mag = output.abs().mean(dim=0).mean(dim=0) # Mean over batch/seq
                    if name not in relevance_scores:
                        relevance_scores[name] = mag.cpu()
                    else:
                        relevance_scores[name] += mag.cpu()
            return fn

        print("Registering stability hooks...")
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(hook_fn(name)))

        samples = [
            "The clinical diagnosis for the patient is",
            "To perform a successful model merge, one must consider",
            "The mathematical foundation of deep learning relies on"
        ]
        
        print(f"Processing {len(samples)} stability samples...")
        for text in samples:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                self.model(**inputs)
        
        for hook in hooks:
            hook.remove()
            
        # Normalize and Broadcast
        final_scores = {}
        for name, score in relevance_scores.items():
            weight_shape = dict(self.model.named_modules())[name].weight.shape
            # score is [OutFeatures]. Reshape to [OutFeatures, 1] then expand to [OutFeatures, InFeatures]
            final_scores[name + ".weight"] = score.view(-1, 1).expand(weight_shape).clone()
            
        return final_scores
