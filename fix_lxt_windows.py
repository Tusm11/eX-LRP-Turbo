import os
import lxt
from pathlib import Path

def fix_lxt():
    lxt_path = Path(lxt.__file__).parent
    models_init = lxt_path / "efficient" / "models" / "__init__.py"
    
    if not models_init.exists():
        print(f"Could not find {models_init}")
        return

    print(f"Applying Smart Patch to {models_init}...")
    
    # Futuristic models that are causing issues
    ghost_models = ["qwen3", "gemma3", "qwen2_moe", "llama3", "falcon3"]
    
    with open(models_init, "r") as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        # If the line contains a ghost model, comment it out
        if any(ghost in line for ghost in ghost_models):
            if not line.strip().startswith("#"):
                print(f"  Silencing: {line.strip()}")
                new_lines.append(f"# {line}")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
            
    with open(models_init, "w") as f:
        f.writelines(new_lines)
    
    print("✓ LXT Smart Patch applied!")

if __name__ == "__main__":
    fix_lxt()
