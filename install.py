import os
import sys
import subprocess

def install():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        print("Installing ComfyUI-A2SB requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])

if __name__ == "__main__":
    install()
