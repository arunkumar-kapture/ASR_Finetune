import os
from dotenv import load_dotenv
from unsloth import FastModel
from huggingface_hub import login

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

model, processor = FastModel.from_pretrained(
    "unsloth/gemma-4-e4b-it-unsloth-bnb-4bit",
    load_in_4bit=False,
)

model.load_adapter("ArunK-2003/Gemma4FT_v0")

model.save_pretrained_merged(
    "gemma4-asr-tamil-merged",
    processor,
    save_method="merged_16bit",
)

model.push_to_hub_merged(
    "ArunK-2003/Gemma4FT_v0_merged",
    processor,
    save_method="merged_16bit",
    token=os.getenv("HF_TOKEN"),
)