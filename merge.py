import os
import shutil
import subprocess
from dotenv import load_dotenv
from huggingface_hub import login, HfApi
from peft import PeftModel
from transformers import Gemma4ForConditionalGeneration, AutoProcessor
import torch
import torch.nn as nn

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

MERGED_DIR  = "gemma4-asr-tamil-merged"
ADAPTER_DIR = "./gemma4-asr-tamil-v1"
HF_REPO     = "ArunK-2003/Gemma4_Tamil_FT"
BASE_MODEL  = "google/gemma-4-e4b-it"

shutil.rmtree(MERGED_DIR, ignore_errors=True)

print("Loading base model...")
model = Gemma4ForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(BASE_MODEL)

print("Unwrapping Gemma4ClippableLinear layers...")
for name, module in model.named_modules():
    for child_name, child in list(module.named_children()):
        if type(child).__name__ == "Gemma4ClippableLinear":
            setattr(module, child_name, child.linear)

print("Loading adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_DIR)

print("Merging adapter into base model...")
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained(MERGED_DIR, safe_serialization=True)
processor.save_pretrained(MERGED_DIR)

print("Files after merge:")
subprocess.run(["ls", "-lh", MERGED_DIR])

print("Creating HF repo if not exists...")
api = HfApi()
api.create_repo(
    repo_id=HF_REPO,
    repo_type="model",
    exist_ok=True,
    token=os.getenv("HF_TOKEN"),
)

print(f"Uploading to {HF_REPO}...")
api.upload_folder(
    folder_path=MERGED_DIR,
    repo_id=HF_REPO,
    repo_type="model",
    token=os.getenv("HF_TOKEN"),
)
print(f"Done! Model pushed to https://huggingface.co/{HF_REPO}")