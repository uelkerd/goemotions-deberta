#!/usr/bin/env python3
"""
VS Code Jupyter Kernel Setup Script
Ensures all Python environments are properly registered as Jupyter kernels for VS Code
"""

import subprocess
import os
import sys

def install_kernel(python_path, kernel_name, display_name):
    """Install a Jupyter kernel for VS Code"""
    try:
        cmd = [python_path, "-m", "ipykernel", "install", "--user", 
               "--name", kernel_name, "--display-name", display_name]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {display_name} kernel installed successfully")
        else:
            print(f"❌ Failed to install {display_name}: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error installing {display_name}: {e}")

def main():
    """Setup all kernels for the GoEmotions DeBERTa project"""
    print("🔧 Setting up VS Code Jupyter kernels...")
    print("=" * 50)
    
    # Define all environments
    environments = [
        {
            "python_path": "/venv/deberta-v3/bin/python3",
            "kernel_name": "deberta-v3-vscode",
            "display_name": "Python (DeBERTa-v3-VSCode)"
        },
        {
            "python_path": "/venv/main/bin/python3", 
            "kernel_name": "main-vscode",
            "display_name": "Python (Main-VSCode)"
        }
    ]
    
    # Install kernels for each environment
    for env in environments:
        if os.path.exists(env["python_path"]):
            install_kernel(env["python_path"], env["kernel_name"], env["display_name"])
        else:
            print(f"⚠️  Python path not found: {env['python_path']}")
    
    print("\n🎉 Kernel setup complete!")
    print("💡 Restart VS Code to see all kernels in the dropdown")

if __name__ == "__main__":
    main()
