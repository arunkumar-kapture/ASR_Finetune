import os
from huggingface_hub import login
from unsloth import FastModel
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

BASE_MODEL = "unsloth/gemma-4-e4b-it-unsloth-bnb-4bit"
ADAPTER_REPO = "ArunK-2003/Gemma4FT_v0"

MERGED_DIR = "./merged_model"
HF_MERGED_REPO = "ArunK-2003/Gemma4_Tamil"

login(token=HF_TOKEN)

model, tokenizer = FastModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=8192,
    load_in_4bit=True,
)

model.load_adapter(ADAPTER_REPO)
os.makedirs(MERGED_DIR, exist_ok=True)

print("Saving merged 16-bit model locally...")
model.save_pretrained_merged(
    MERGED_DIR,
    tokenizer,
    save_method="merged_16bit"
)
print("Local merged model saved.")


print("Pushing merged model to HF...")
model.push_to_hub_merged(
    HF_MERGED_REPO,
    tokenizer,
    save_method="merged_16bit",
    maximum_memory_usage=0.5,
    token=HF_TOKEN,
)
print(f"Done: https://huggingface.co/{HF_MERGED_REPO}")