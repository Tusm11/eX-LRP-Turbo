import argparse
import os
import subprocess
import sys
import yaml
import json
from pathlib import Path

# Windows Stability Flags
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONUTF8"] = "1"

def run_step(mode, model_path, output_path, hero_threshold=0.01):
    script_path = os.path.join(os.path.dirname(__file__), "lrp_turbo_experimental.py")
    cmd = [sys.executable, script_path, mode, model_path, output_path, "--hero-threshold", str(hero_threshold)]
    print(f"\n>> Launching {mode.upper()} process...")
    subprocess.run(cmd, check=True, env=os.environ.copy())

def main():
    parser = argparse.ArgumentParser(description="TurboQuant LRP Merge Pipeline")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--model1", required=True)
    parser.add_argument("--model2", required=True)
    parser.add_argument("--hero-threshold", type=float, default=0.01)
    parser.add_argument("--density", type=float, default=0.9) # Increased for High-Fidelity
    parser.add_argument("--output", required=True)
    
    args = parser.parse_args()
    work_dir = Path("./turbo_work")
    work_dir.mkdir(parents=True, exist_ok=True)
    
    m1_out = work_dir / "model1"
    m1_out.mkdir(parents=True, exist_ok=True)
    if not (m1_out / "turbo_quant_model.safetensors").exists():
        run_step("audit", args.model1, str(m1_out), args.hero_threshold)
        run_step("compress", args.model1, str(m1_out), args.hero_threshold)
    
    m2_out = work_dir / "model2"
    m2_out.mkdir(parents=True, exist_ok=True)
    if args.model1 == args.model2:
        import shutil
        for f in ["lrp_scores.safetensors", "turbo_quant_model.safetensors"]:
            if (m1_out / f).exists(): shutil.copy(m1_out / f, m2_out / f)
    else:
        if not (m2_out / "turbo_quant_model.safetensors").exists():
            run_step("audit", args.model2, str(m2_out), args.hero_threshold)
            run_step("compress", args.model2, str(m2_out), args.hero_threshold)
        
    manifest = {
        args.model1: {
            "lrp_scores": str(m1_out / "lrp_scores.safetensors"),
            "turbo_model": str(m1_out / "turbo_quant_model.safetensors")
        },
        args.model2: {
            "lrp_scores": str(m2_out / "lrp_scores.safetensors"),
            "turbo_model": str(m2_out / "turbo_quant_model.safetensors")
        }
    }
    with open(work_dir / "turbo_manifest.json", "w") as f:
        json.dump(manifest, f)
    
    print("\n--- Starting High-Fidelity Final Merge ---")
    config = {
        "merge_method": "lrp_turbo",
        "base_model": args.base_model,
        "parameters": {"density": args.density}, # Use the new density
        "models": [
            {"model": args.model1, "parameters": {"weight": 1.0}},
            {"model": args.model2, "parameters": {"weight": 1.0}}
        ],
        "output_path": args.output,
        "dtype": "float16",
    }
    
    config_path = work_dir / "merge_config.yml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
        
    subprocess.run([sys.executable, "register_turbo.py", str(config_path), args.output], check=True, env=os.environ.copy())
    print(f"\n✨ SUCCESS! High-Fidelity Merged model: {args.output}")

if __name__ == "__main__":
    main()
