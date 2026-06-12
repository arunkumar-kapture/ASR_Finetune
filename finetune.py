import os
import gc
import json
import random
import shutil
import time
import soundfile as sf
import numpy as np
import torch
import psutil
import jiwer
import wandb
from datasets import load_dataset, Audio
from itertools import islice
from unsloth import FastModel
from unsloth.trainer import UnslothVisionDataCollator
from transformers import TrainerCallback, TextStreamer
from trl import SFTTrainer, SFTConfig
from typing import List, Dict
from tqdm.auto import tqdm
from dotenv import load_dotenv
load_dotenv()

print("Setting up the project...\n")
try:
    from huggingface_hub import login
    login(token=os.getenv("HF_TOKEN"))
except Exception as e:
    print(f"HF login error: {e}")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

wandb.login(key=os.getenv("WB_API_KEY"))

MODEL_PATH     = "unsloth/gemma-4-E4B-it"
RUN_NAME       = "gemma4-asr-tamil-v1"
LORA_PATH      = f"./{RUN_NAME}"
TARGET_SR      = 16000
MIN_AUDIO_LEN  = 2.0
MAX_AUDIO_LEN  = 30.0
SHUFFLE_BUFFER = 500

TRAIN_SAMPLES: Dict[str, int] = {
    "tamil":   18500,
    "english": 0,
}
EVAL_SAMPLES: Dict[str, int] = {
    "tamil":   400,
    "english": 0,
}

use_bf16 = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability(0)[0] >= 8
)

gpu_vram_gb = (
    torch.cuda.get_device_properties(0).total_memory / 1024**3
    if torch.cuda.is_available() else 0
)

print(f"GPU VRAM : {gpu_vram_gb:.1f} GB")
print(f"BF16     : {use_bf16}\n")

print("Loading the model...\n")
model, processor = FastModel.from_pretrained(
    MODEL_PATH,
    dtype=None,
    max_seq_length=8192,
    load_in_4bit=True,
    full_finetuning=False,
)
 
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "post", "linear_start", "linear_end",
        "embedding_projection",
        "ffw_layer_1", "ffw_layer_2",
        "output_proj",
    ]
)
model.print_trainable_parameters()

DISK_ROOT       = "./asr_data"
TRAIN_AUDIO_DIR = os.path.join(DISK_ROOT, "train_wavs")
TRAIN_JSONL     = os.path.join(DISK_ROOT, "train.jsonl")
EVAL_AUDIO_DIR  = os.path.join(DISK_ROOT, "eval_wavs")
EVAL_JSONL      = os.path.join(DISK_ROOT, "eval.jsonl")

os.makedirs(TRAIN_AUDIO_DIR, exist_ok=True)
os.makedirs(EVAL_AUDIO_DIR,  exist_ok=True)

for _p in [TRAIN_JSONL, EVAL_JSONL]:
    if os.path.exists(_p):
        os.remove(_p)

INSTRUCTION = "Transcribe the following audio accurately. Return only the transcription text, nothing else."

def log_ram(tag: str = "") -> None:
    vm = psutil.virtual_memory()
    print(f"[RAM {tag}] {vm.used/1024**3:.2f} / {vm.total/1024**3:.2f} GB used")


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def decode_audio(audio_info):
    if isinstance(audio_info, dict):
        return audio_info["array"], audio_info["sampling_rate"]
    elif hasattr(audio_info, "get_all_samples"):
        audio_data = audio_info.get_all_samples()
        array = audio_data.data
        sr = audio_data.sample_rate
        if hasattr(array, "shape") and len(array.shape) > 1:
            array = array[0]
        return array, sr
    else:
        raise ValueError(f"Unknown audio format: {type(audio_info)}")


def safe_write_wav(path: str, array: np.ndarray, sr: int, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            sf.write(path, array, sr)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return True
            raise IOError("Empty file after write")
        except Exception as e:
            if os.path.exists(path):
                os.remove(path)
            if attempt < retries - 1:
                time.sleep(0.5)
            else:
                print(f"[ERROR] Write failed for {path}: {e}")
    return False

def process_stream_to_disk(
    dataset,
    lang: str,
    n_samples: int,
    audio_dir: str,
    tag: str,
    prefix: str,
    shuffle_buffer: int = SHUFFLE_BUFFER,
    skip_n: int = 0,
    stream_multiplier: int = 2,
) -> List[Dict]:
    entries: List[Dict] = []
    stats = {
        "no_audio": 0,
        "empty_text": 0,
        "decode_fail": 0,
        "empty_array": 0,
        "duration": 0,
        "write_fail": 0,
        "success": 0,
    }
    if shuffle_buffer > 0 and n_samples > 10:
        dataset = dataset.shuffle(seed=42, buffer_size=shuffle_buffer)

    dataset = dataset.cast_column("audio", Audio(sampling_rate=TARGET_SR))

    log_ram(f"before {lang} {tag}")
    stream = islice(iter(dataset), skip_n, skip_n + n_samples * stream_multiplier)
    pbar = tqdm(stream, desc=f"{lang} {tag}", total=n_samples)

    for i, sample in enumerate(pbar):
        if len(entries) >= n_samples:
            break
        try:
            audio_info = sample.get("audio") or sample.get("audio_filepath")
            if audio_info is None:
                stats["no_audio"] += 1
                continue
            text = sample.get("text", "").strip()
            if not text or "<unintelligible>" in text:
                stats["empty_text"] += 1
                continue

            try:
                array, sr = decode_audio(audio_info)
            except Exception as e:
                if i < 3:
                    print(f"decode error: {e}")
                stats["decode_fail"] += 1
                continue

            if array is None or len(array) == 0:
                stats["empty_array"] += 1
                continue

            duration = len(array) / sr
            if not (MIN_AUDIO_LEN <= duration <= MAX_AUDIO_LEN):
                stats["duration"] += 1
                continue

            if array.ndim == 2:
                array = array.mean(axis=0)
              
            wav_name = f"{prefix}_{tag}_{len(entries):06d}.wav"
            wav_path = os.path.join(audio_dir, wav_name)

            if not safe_write_wav(wav_path, array, sr):
                stats["write_fail"] += 1
                continue

            entries.append({"audio": wav_path, "text": text, "lang": lang.capitalize()})
            stats["success"] += 1
            pbar.set_postfix(saved=len(entries), skipped=i - len(entries))
            del array

        except Exception as e:
            print(f"[FATAL SAMPLE ERROR] {e}")
            continue

    pbar.close()
    free_memory()

    print(f"\n==== DEBUG STATS ({lang} {tag}) ====")
    for k, v in stats.items():
        print(f"{k}: {v}")

    if stats["success"] < n_samples * 0.8:
        print(
            f"[WARNING] Only collected {stats['success']} / {n_samples} samples for {lang} {tag}. "
            f"Consider increasing stream_multiplier (currently {stream_multiplier})."
        )

    log_ram(f"after {lang} {tag} ({len(entries)} saved)")
    return entries


def save_jsonl(entries: List[Dict], path: str, mode: str = "a") -> None:
    with open(path, mode, encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print("Generating the dataset...\n")

ds_ta_train = load_dataset(
    "ai4bharat/Kathbath", "tamil",
    split="train",
    streaming=True
)
ds_ta_eval = load_dataset(
    "ai4bharat/Kathbath", "tamil",
    split="valid",
    streaming=True,
    trust_remote_code=True,
)

train_entries: List[Dict] = []
eval_entries:  List[Dict] = [] 
ta_train = process_stream_to_disk(
    ds_ta_train, "tamil", TRAIN_SAMPLES["tamil"],
    TRAIN_AUDIO_DIR, "train", "ta"
)
save_jsonl(ta_train, TRAIN_JSONL, mode="w")
train_entries.extend(ta_train)
del ta_train, ds_ta_train
free_memory()

if EVAL_SAMPLES["tamil"] > 0:
    ta_eval = process_stream_to_disk(
        ds_ta_eval, "tamil", EVAL_SAMPLES["tamil"],
        EVAL_AUDIO_DIR, "eval", "ta", shuffle_buffer=0
    )
    save_jsonl(ta_eval, EVAL_JSONL, mode="w")
    eval_entries.extend(ta_eval)
    del ta_eval
free_memory()
del ds_ta_eval

if TRAIN_SAMPLES["english"] > 0:
    en_train = process_stream_to_disk(
        load_dataset("ai4bharat/Svarah", split="test", streaming=True),
        "english", TRAIN_SAMPLES["english"],
        TRAIN_AUDIO_DIR, "train", "en",
        skip_n=0,
    )
    save_jsonl(en_train, TRAIN_JSONL, mode="a")
    train_entries.extend(en_train)
    del en_train
    free_memory()

random.shuffle(train_entries)
eval_data = eval_entries

print(f"Train size : {len(train_entries)}")
print(f"Eval  size : {len(eval_data)}")
log_ram("After dataset build")

def format_entries_to_messages(entries: List[Dict]) -> List[Dict]:
    formatted = []
    for entry in entries:
        audio_array, _ = sf.read(entry["audio"], dtype="float32")
        formatted.append({
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are an assistant that transcribes speech accurately.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_array},
                        {"type": "text", "text": f"Language: {entry['lang']}\n{INSTRUCTION}"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": entry["text"]}],
                },
            ]
        })
    return formatted
 
 
print("Formatting datasets into message format...")
converted_train = format_entries_to_messages(train_entries)
converted_eval  = format_entries_to_messages(eval_data)
free_memory()
 
from datasets import Dataset
converted_train = Dataset.from_list(converted_train)
converted_eval  = Dataset.from_list(converted_eval)
 
 
class MakeEveryCheckpointInferableCallback(TrainerCallback):
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path
 
    def on_save(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return control
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        copied, skipped = [], []
        for fn in [
            "config.json", "generation_config.json", "preprocessor_config.json",
            "processor_config.json", "tokenizer_config.json", "tokenizer.json",
            "special_tokens_map.json", "chat_template.json", "merges.txt", "vocab.json",
        ]:
            src = os.path.join(self.base_model_path, fn)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(ckpt_dir, fn))
                copied.append(fn)
            else:
                skipped.append(fn)
        print(f"[Checkpoint {state.global_step}] Copied: {copied}")
        if skipped:
            print(f"[Checkpoint {state.global_step}] Skipped (not found): {skipped}")
        return control
 
class WandbMetricsCallback(TrainerCallback):
    def __init__(self, processor, eval_entries: List[Dict], sample_size: int = 50):
        self.processor    = processor
        self.eval_entries = eval_entries
        self.sample_size  = sample_size

    @staticmethod
    def get_memory_stats() -> Dict:
        stats = {}
        if torch.cuda.is_available():
            stats["gpu/allocated_gb"]     = torch.cuda.memory_allocated()     / 1024**3
            stats["gpu/reserved_gb"]      = torch.cuda.memory_reserved()      / 1024**3
            stats["gpu/max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
        vm = psutil.virtual_memory()
        stats["cpu/used_gb"]  = vm.used  / 1024**3
        stats["cpu/total_gb"] = vm.total / 1024**3
        return stats

    def compute_lang_wer(self, model) -> Dict:
        if not self.eval_entries:
            return {}

        preds: Dict[str, List] = {}
        refs:  Dict[str, List] = {}

        FastModel.for_inference(model)

        n = min(self.sample_size, len(self.eval_entries))
        for entry in self.eval_entries[:n]:
            lang = entry["lang"]
            ref  = entry["text"]
            audio_array, _ = sf.read(entry["audio"], dtype="float32")

            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are an assistant that transcribes speech accurately.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_array},
                        {"type": "text", "text": f"Language: {lang}\n{INSTRUCTION}"},
                    ],
                },
            ]
 
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to("cuda")

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    use_cache=True,
                    do_sample=False,
                )

            input_len = inputs["input_ids"].shape[-1]
            result = processor.decode(out_ids[0][input_len:], skip_special_tokens=True)
            preds.setdefault(lang, []).append(result)
            refs.setdefault(lang,  []).append(ref)
 
            del audio_array, inputs, out_ids
            free_memory()
 
        FastModel.for_training(model)
        free_memory()
 
        wer_results = {}
        for lang in preds:
            wer_val = jiwer.wer(refs[lang], preds[lang])
            wer_results[f"eval/wer_{lang}"] = wer_val
            print(f"[WER] {lang}: {wer_val:.4f}")
        return wer_results
 
    def on_step_end(self, args, state, control, **kwargs):
        wandb.log(self.get_memory_stats(), step=state.global_step)
 
    def on_epoch_end(self, args, state, control, **kwargs):
        lang_metrics = self.compute_lang_wer(kwargs["model"])
        wandb.log({**lang_metrics, **self.get_memory_stats()}, step=state.global_step)
 
 
wandb.init(
    project="gemma4-asr-ft",
    name=RUN_NAME,
    config={
        "model":                       MODEL_PATH,
        "gpu_vram_gb":                 gpu_vram_gb,
        "learning_rate":               2e-4,
        "lr_scheduler_type":           "cosine",
        "warmup_ratio":                0.03,
        "per_device_train_batch_size": 3,
        "gradient_accumulation_steps": 5,
        "num_train_epochs":            1,
        "logging_steps":               50,
        "save_steps":                  300,
        "eval_steps":                  300,
        "save_total_limit":            4,
        "bf16":                        use_bf16,
        "lora_r":                      32,
        "lora_alpha":                  64,
        "lora_dropout":                0.1,
        "train_samples":               len(train_entries),
        "train_samples_cfg":           TRAIN_SAMPLES,
        "eval_samples":                len(eval_data),
    },
)

trainer = SFTTrainer(
    model=model,
    train_dataset=converted_train,
    eval_dataset=converted_eval,
    processing_class=processor.tokenizer,
    data_collator=UnslothVisionDataCollator(model, processor),
    args=SFTConfig(
        output_dir=LORA_PATH,

        num_train_epochs=1,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=5,
        warmup_ratio=0.03,
        learning_rate=2e-4,

        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.001,

        logging_steps=50,
        logging_first_step=True,

        eval_strategy="steps",
        eval_steps=300,

        save_strategy="steps",
        save_steps=300,
        save_total_limit=4,

        bf16=use_bf16,
        fp16=not use_bf16,

        dataloader_num_workers=0,
        dataloader_pin_memory=False,

        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=8192,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        report_to="wandb",

        push_to_hub=True,
        hub_model_id="ArunK-2003/Gemma4FT_v0",
        hub_strategy="every_save",
    ),
    callbacks=[
        MakeEveryCheckpointInferableCallback(base_model_path=MODEL_PATH),
        WandbMetricsCallback(
            processor=processor,
            eval_entries=eval_data,
            sample_size=len(eval_data),
        ),
    ],
)

print("Started training...\n")
trainer.train()

print("Completed training and saving the adaptor...\n")
if trainer.args.process_index == 0:
    final_adapter_dir = os.path.join(LORA_PATH, "final_lora_adapter")
    model.save_pretrained(final_adapter_dir)
    processor.tokenizer.save_pretrained(final_adapter_dir)
    print(f"Final LoRA adapter saved to {final_adapter_dir}")

wandb.finish()