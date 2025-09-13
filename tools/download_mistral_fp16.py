#!/usr/bin/env python3
"""Download unquantized Mistral-7B model for inference engine."""

import os
import sys
from pathlib import Path

def main():
    print("🚀 Downloading Unquantized Mistral-7B Model")
    print("=" * 50)
    
    print("""
The current Mistral model you have is 4-bit quantized, which requires 
special dequantization logic that your inference engine doesn't support yet.

To run inference properly, you need an unquantized (FP16/FP32) version.

RECOMMENDED OPTIONS:

1. Mistral-7B-Instruct-v0.3 (FP16) - Hugging Face:
   huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --cache-dir ~/.cache/huggingface

2. Mistral-7B-v0.1 (Base model, FP16):
   huggingface-cli download mistralai/Mistral-7B-v0.1 --cache-dir ~/.cache/huggingface

3. Convert existing quantized model to FP16 (advanced):
   - Use transformers library to load and save in FP16 format

INSTALL REQUIREMENTS:
   pip install transformers torch safetensors huggingface_hub

EXAMPLE DOWNLOAD COMMAND:
   huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 \\
       --cache-dir ~/.cache/huggingface \\
       --include "*.safetensors" "*.json"

This will download ~14GB of unquantized weights that will work directly 
with your inference engine!
""")

    print("\n" + "=" * 50)
    print("After downloading, use the new path with your inference engine:")
    print("./build/run_mistral_inference <new_model_path> \"Hello!\"")

if __name__ == "__main__":
    main()
