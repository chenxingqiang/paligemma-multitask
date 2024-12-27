import os
import sys
import subprocess

def setup_environment():
    """Setup environment for PaliGemma including big_vision repository."""
    current_dir = os.getcwd()
    big_vision_path = os.path.join(current_dir, "big_vision_repo")
    
    # Add big_vision_repo to PYTHONPATH
    os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + os.pathsep + big_vision_path
    sys.path.append(big_vision_path)
    
    # Clone big_vision repository if not exists
    if not os.path.exists("big_vision_repo"):
        subprocess.run(
            ["git", "clone", "--quiet", "--branch=main", "--depth=1", 
             "https://github.com/google-research/big_vision", "big_vision_repo"],
            check=True
        )