import torch
import torch.nn as nn
import os
import sys
import gc
import argparse
from pathlib import Path
from safetensors.torch import save_file, load_file
from lrp_computer import LRPComputer

class TurboLRPComputer:
    def __init__(self, model_path, device="cpu"):
        self.computer = LRPComputer(model_path, device=device)
        self.device = device

    def audit_model(self, output_dir):
        """Compute LRP scores and save as safetensors with RAM optimization."""
        scores = self.computer.compute_all_relevance_scores()
        
        # --- PRODUCTION RAM PATCH ---
        # Move to CPU and cast to half precision layer-by-layer to avoid spikes
        compressed_scores = {}
        for k, v in scores.items():
            compressed_scores[k] = v.detach().cpu().half()
        
        # Free up the original scores dict immediately
        del scores
        gc.collect()
        
        save_file(compressed_scores, os.path.join(output_dir, "lrp_scores.safetensors"))
        print(f"✓ LRP scores saved to {output_dir}/lrp_scores.safetensors")

    def apply_turbo_quant(self, lrp_scores_path, output_dir, hero_threshold=0.01):
        """Compress model using LRP scores and save as safetensors."""
        scores = load_file(lrp_scores_path)
        model = self.computer.model
        compressed_model = {}

        print(f"Compressing with hero_threshold={hero_threshold}...")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data.to(self.device)
                importance = scores.get(name + ".weight")
                if importance is None: continue
                
                importance = importance.to(self.device)
                hero_mask = importance > (importance.max() * hero_threshold)
                hero_weights = weight[hero_mask]
                noise_weights = weight[~hero_mask]
                
                layer_data = {
                    "hero_weights": hero_weights.detach().cpu().half(),
                    "hero_mask": hero_mask.detach().cpu().bool(),
                    "shape": torch.tensor(weight.shape, dtype=torch.int32)
                }

                if noise_weights.numel() > 0:
                    noise_flat = noise_weights.flatten()
                    block_size = 1024
                    padding = (block_size - (noise_flat.numel() % block_size)) % block_size
                    noise_padded = torch.cat([noise_flat, torch.zeros(padding, device=self.device)])
                    
                    torch.manual_seed(42)
                    R = torch.randn(block_size, block_size, device=self.device)
                    Q, _ = torch.linalg.qr(R)
                    
                    noise_blocks = noise_padded.view(-1, block_size)
                    rotated_blocks = torch.matmul(noise_blocks, Q)
                    
                    scale = rotated_blocks.abs().max()
                    noise_quant = (rotated_blocks / (scale + 1e-8) * 127).to(torch.int8)
                    
                    layer_data["noise_quant"] = noise_quant.detach().cpu()
                    layer_data["noise_scales"] = torch.tensor([scale], dtype=torch.float16)
                
                compressed_model[name] = layer_data
                del weight, importance, hero_mask, noise_weights
                gc.collect()

        # Flatten for Safetensors and apply RAM patch to compression too
        flat_save = {}
        for name, data in compressed_model.items():
            for key, val in data.items():
                flat_save[f"{name}.{key}"] = val

        del compressed_model
        gc.collect()

        save_file(flat_save, os.path.join(output_dir, "turbo_quant_model.safetensors"))
        print(f"✓ Compressed model saved to {output_dir}/turbo_quant_model.safetensors")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["audit", "compress"])
    parser.add_argument("model_path")
    parser.add_argument("output_dir")
    parser.add_argument("--hero-threshold", type=float, default=0.01)
    args = parser.parse_args()

    computer = TurboLRPComputer(args.model_path)
    if args.mode == "audit":
        computer.audit_model(args.output_dir)
    else:
        scores_path = os.path.join(args.output_dir, "lrp_scores.safetensors")
        computer.apply_turbo_quant(scores_path, args.output_dir, args.hero_threshold)

if __name__ == "__main__":
    main()
