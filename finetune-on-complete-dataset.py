#!/usr/bin/env python3
"""
Production-Ready Fine-Tuning Script for Qwen-VL Models (e.g., bone fracture detection)
- Lazy image loading (memory efficient)
- Fully configurable via top-level CONFIG
- Disables problematic Torch compile
- Handles common errors gracefully

✅ To use:
  1. Set your paths & hyperparameters in CONFIG below
  2. Ensure dataset format: {"image_path": str, "text_input": str, "target_output": str}
  3. Run: python fine_qwen_vl.py
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

# ⚠️ Critical: Disable TorchDynamo to avoid GCC compile errors on some systems
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
import gc
from PIL import Image
import json
from typing import List, Dict, Any
from datasets import DatasetDict
import warnings
warnings.filterwarnings("ignore")

from unsloth import FastVisionModel
from transformers import TrainingArguments, Trainer

# =============================================================================
# 🔧 USER CONFIGURATION — EDIT ONLY THIS BLOCK
# =============================================================================
CONFIG = {
    # Model
    "model_name": "Qwen/Qwen3-VL-2B-Instruct",
    
    # Dataset
    "dataset_path": "./xr_bones_qwen_format/hf_dataset_fixed",  # HuggingFace DatasetDict path
    "eval_sample_size": 100,  # Use subset for eval to save time/memory

    # Image
    "image_size": (512, 512),  # (width, height)

    # Training
    "output_dir": "./qwen3vl-xray-full",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "max_seq_length": 512,

    # LoRA
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,

    # System
    "seed": 3407,
    "dataloader_num_workers": 0,  # Keep 0 to avoid fork issues with PIL + lazy loading
}

# =============================================================================
# DO NOT EDIT BELOW THIS LINE (unless you know what you're doing)
# =============================================================================

print("\n" + "="*60)
print("MEMORY-EFFICIENT FULL TRAINING (PRODUCTION READY)")
print("="*60)

# =============================================================================
# LAZY-LOADING DATASET CLASS
# =============================================================================
class LazyImageDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, img_size):
        self.dataset = hf_dataset
        self.img_size = img_size
        print(f"✅ Dataset wrapper created for {len(hf_dataset)} samples")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img_path = sample['image_path']
        try:
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
                image = image.resize(self.img_size, Image.LANCZOS)
            else:
                image = Image.new('RGB', self.img_size, color='white')
        except Exception as e:
            print(f"⚠️  Error loading {img_path}: {e}")
            image = Image.new('RGB', self.img_size, color='white')
        
        return {
            "image": image,
            "question": sample.get('text_input', 'Analyze this X-ray.'),
            "answer": sample.get('target_output', 'Normal')
        }

# =============================================================================
# DATA COLLATOR
# =============================================================================
class QwenVisionCollator:
    def __init__(self, processor):
        self.processor = processor
        
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        processed_samples = []
        for item in batch:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["image"]},
                        {"type": "text", "text": item["question"]}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": item["answer"]}]
                }
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            processed_samples.append({"text": text, "image": item["image"]})
        
        texts = [s["text"] for s in processed_samples]
        images = [s["image"] for s in processed_samples]
        
        inputs = self.processor(
            text=texts, images=images, return_tensors="pt",
            padding=True, truncation=True, max_length=CONFIG["max_seq_length"]
        )
        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100
        inputs["labels"] = labels
        return inputs

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# =============================================================================
# SETUP
# =============================================================================
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM: {vram:.1f} GB")
cleanup()

# =============================================================================
# LOAD MODEL
# =============================================================================
print("\n📦 Loading model...")

model, processor = FastVisionModel.from_pretrained(
    CONFIG["model_name"],
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
    max_seq_length=CONFIG["max_seq_length"],
)
print("✅ Model loaded")

# Configure LoRA
FastVisionModel.for_training(model)
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=CONFIG["lora_dropout"],
    bias="none",
    random_state=CONFIG["seed"],
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"✅ Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
cleanup()

# =============================================================================
# LOAD DATASET
# =============================================================================
print("\n📂 Loading dataset (lazy mode)...")
dataset = DatasetDict.load_from_disk(CONFIG["dataset_path"])
print(f"✅ Found: Train={len(dataset['train'])}, Val={len(dataset['validation'])}")

train_dataset = LazyImageDataset(dataset['train'], CONFIG["image_size"])
eval_dataset = LazyImageDataset(
    dataset['validation'].select(range(min(CONFIG["eval_sample_size"], len(dataset['validation'])))),
    CONFIG["image_size"]
)
print(f"✅ Ready: Train={len(train_dataset)}, Eval={len(eval_dataset)}")
cleanup()

# =============================================================================
# TEST COLLATOR
# =============================================================================
print("\n🧪 Testing data pipeline...")
collator = QwenVisionCollator(processor)
test_output = collator([train_dataset[0]])
print("✅ Data pipeline working!")
print(f"  Input shape: {test_output['input_ids'].shape}")
print(f"  Pixel shape: {test_output['pixel_values'].shape}")
cleanup()

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
total_samples = len(train_dataset)
effective_batch = CONFIG["per_device_train_batch_size"] * CONFIG["gradient_accumulation_steps"]
steps_per_epoch = total_samples // effective_batch

training_args = TrainingArguments(
    output_dir=CONFIG["output_dir"],
    per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
    per_device_eval_batch_size=CONFIG["per_device_eval_batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    num_train_epochs=CONFIG["num_train_epochs"],
    learning_rate=CONFIG["learning_rate"],
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    optim="paged_adamw_8bit",
    eval_strategy="steps",
    eval_steps=max(steps_per_epoch // 4, 50),
    save_strategy="steps",
    save_steps=max(steps_per_epoch // 2, 100),
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_steps=50,
    logging_first_step=True,
    report_to="none",
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=CONFIG["dataloader_num_workers"],
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    seed=CONFIG["seed"],
    data_seed=CONFIG["seed"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
    tokenizer=processor,
)
print("✅ Trainer configured!")

# =============================================================================
# TRAIN
# =============================================================================
print("\n" + "="*60)
print("🚀 STARTING TRAINING")
print("="*60)
print(f"⏰ Started at: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60 + "\n")

final_save_dir = os.path.join(CONFIG["output_dir"] + "-final")
try:
    train_result = trainer.train()
    
    print("\n✅ TRAINING COMPLETE!")
    os.makedirs(final_save_dir, exist_ok=True)
    model.save_pretrained(final_save_dir)
    processor.save_pretrained(final_save_dir)
    
    # Save config + metrics
    info = {
        "config": CONFIG,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "trainable_params": trainable,
        "total_params": total,
        "metrics": train_result.metrics,
    }
    with open(os.path.join(final_save_dir, "training_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"✅ Final model saved to: {final_save_dir}")

except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    fallback_dir = CONFIG["output_dir"] + "-error"
    try:
        model.save_pretrained(fallback_dir)
        processor.save_pretrained(fallback_dir)
        print(f"✅ Partial model saved to: {fallback_dir}")
    except:
        print("❌ Could not save partial model")
    exit(1)

finally:
    cleanup()

print("\n" + "="*60)
print("🎉 SUCCESS! MODEL READY FOR INFERENCE")
print("="*60)
print(f"➡️  Use: FastVisionModel.from_pretrained('{final_save_dir}')")
print("="*60)