#!/usr/bin/env python3
"""
Qwen3-VL Fine-tuning for Bone Fracture Detection
Uses your preprocessed xr_bones_qwen_format dataset
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import gc
from PIL import Image
import json
from typing import List, Dict, Any
from datasets import Dataset, DatasetDict
import warnings
warnings.filterwarnings("ignore")

from unsloth import FastVisionModel
from transformers import TrainingArguments, Trainer

print("\n" + "="*60)
print("QWEN3-VL BONE FRACTURE FINE-TUNING")
print("="*60)

# =============================================================================
# DATA COLLATOR
# =============================================================================
class QwenVisionCollator:
    """Proper collator for Qwen3-VL"""
    
    def __init__(self, processor):
        self.processor = processor
        
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process batch with proper message format"""
        
        # Process each sample
        processed_samples = []
        for item in batch:
            image = item["image"]
            question = item["question"]
            answer = item["answer"]
            
            # Qwen3-VL message format with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}]
                }
            ]
            
            # Apply chat template - adds image tokens
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            processed_samples.append({"text": text, "image": image})
        
        # Process all together
        texts = [s["text"] for s in processed_samples]
        images = [s["image"] for s in processed_samples]
        
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Create labels
        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100
        inputs["labels"] = labels
        
        # DON'T move to GPU - Trainer handles it
        return inputs

# =============================================================================
# SETUP
# =============================================================================
def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM: {vram:.1f} GB")
cleanup()

# =============================================================================
# LOAD MODEL
# =============================================================================
print("\n📦 Loading Qwen3-VL-2B-Instruct...")

model, processor = FastVisionModel.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
    max_seq_length=512,
)
print("✅ Model loaded")
cleanup()

# Configure LoRA
FastVisionModel.for_training(model)
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    random_state=3407,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"✅ Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
cleanup()

# =============================================================================
# LOAD DATASET
# =============================================================================
print("\n📂 Loading preprocessed dataset...")

dataset_path = "./xr_bones_qwen_format/hf_dataset_fixed"
if not os.path.exists(dataset_path):
    print(f"❌ Dataset not found: {dataset_path}")
    exit(1)

dataset = DatasetDict.load_from_disk(dataset_path)
print(f"✅ Loaded dataset")
print(f"  Train: {len(dataset['train'])} samples")
print(f"  Validation: {len(dataset['validation'])} samples")

# =============================================================================
# PREPARE TRAINING DATA
# =============================================================================
print("\n🔧 Preparing training samples...")

def prepare_samples(split, max_samples=None):
    """Convert dataset to training format"""
    samples = []
    
    num_samples = min(max_samples, len(split)) if max_samples else len(split)
    
    for i in range(num_samples):
        sample = split[i]
        
        # Load image
        img_path = sample['image_path']
        if not os.path.exists(img_path):
            print(f"⚠️  Image not found: {img_path}")
            image = Image.new('RGB', (512, 512), color='white')
        else:
            image = Image.open(img_path).convert('RGB')
            image = image.resize((512, 512), Image.LANCZOS)
        
        # Get text
        question = sample.get('text_input', 'Analyze this X-ray image.')
        answer = sample.get('target_output', 'Normal')
        
        samples.append({
            "image": image,
            "question": question,
            "answer": answer
        })
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{num_samples}...")
    
    return Dataset.from_list(samples)

# Start small for testing
TEST_SAMPLES = 50  # Increase this once it works
print(f"Creating {TEST_SAMPLES} training samples...")

train_dataset = prepare_samples(dataset['train'], max_samples=TEST_SAMPLES)
eval_dataset = prepare_samples(dataset['validation'], max_samples=10)

print(f"✅ Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
cleanup()

# =============================================================================
# TEST COLLATOR
# =============================================================================
print("\n🧪 Testing data collator...")

collator = QwenVisionCollator(processor)

try:
    test_batch = [train_dataset[0]]
    test_output = collator(test_batch)
    
    print("✅ Collator working!")
    print(f"  Input IDs: {test_output['input_ids'].shape}")
    print(f"  Labels: {test_output['labels'].shape}")
    print(f"  Pixel values: {test_output['pixel_values'].shape}")
    print(f"  Device: {test_output['input_ids'].device}")  # Should be CPU
    
    # Check for image tokens
    decoded = processor.decode(test_output['input_ids'][0], skip_special_tokens=False)
    has_tokens = '<|vision_start|>' in decoded or '<|image_pad|>' in decoded
    print(f"  Image tokens: {'✅ Found' if has_tokens else '⚠️  Missing'}")
    
except Exception as e:
    print(f"❌ Collator failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

cleanup()

# =============================================================================
# TRAINING
# =============================================================================
print("\n⚙️  Configuring training...")

training_args = TrainingArguments(
    output_dir="./qwen3vl-xray-model",
    
    # Batch settings
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    
    # Training schedule
    num_train_epochs=3,
    max_steps=-1,  # Use epochs
    
    # Learning
    learning_rate=2e-4,
    warmup_steps=10,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    optim="paged_adamw_8bit",
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=10,
    
    # Saving
    save_strategy="steps",
    save_steps=25,
    save_total_limit=2,
    
    # Logging
    logging_steps=5,
    report_to="none",
    
    # Performance
    fp16=True,
    bf16=False,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,  # Important!
    remove_unused_columns=False,
    gradient_checkpointing=True,
    
    # Seeds
    seed=3407,
    data_seed=3407,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
    tokenizer=processor,
)

print("✅ Trainer ready")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"Total training steps: ~{len(train_dataset) * training_args.num_train_epochs // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)}")

# =============================================================================
# TRAIN
# =============================================================================
print("\n" + "="*60)
print("🚀 STARTING TRAINING")
print("="*60 + "\n")

try:
    train_result = trainer.train()
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"Final loss: {train_result.metrics.get('train_loss', 'N/A')}")
    
    # Save model
    print("\n💾 Saving model...")
    save_dir = "./qwen3vl-xray-final"
    os.makedirs(save_dir, exist_ok=True)
    
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    
    # Save info
    info = {
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "trainable_params": trainable,
        "metrics": train_result.metrics,
        "model": "Qwen3-VL-2B-Instruct",
        "task": "bone_fracture_detection"
    }
    
    with open(f"{save_dir}/training_info.json", "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"✅ Model saved to {save_dir}/")
    
    # Inference example
    print("\n" + "="*60)
    print("INFERENCE EXAMPLE")
    print("="*60)
    print("""
To use your fine-tuned model:

from unsloth import FastVisionModel
from PIL import Image

# Load model
model, processor = FastVisionModel.from_pretrained(
    "./qwen3vl-xray-final",
    load_in_4bit=True,
)

# Prepare input
image = Image.open("xray.jpg")
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Analyze this X-ray for fractures."}
    ]
}]

# Generate
inputs = processor.apply_chat_template(
    messages, 
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=128)
response = processor.decode(outputs[0], skip_special_tokens=True)
print(response)
""")

except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted by user")
    
except Exception as e:
    print(f"\n\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    
    # Save partial
    try:
        print("\n💾 Saving partial model...")
        os.makedirs("./qwen3vl-xray-partial", exist_ok=True)
        model.save_pretrained("./qwen3vl-xray-partial")
        processor.save_pretrained("./qwen3vl-xray-partial")
        print("✅ Partial model saved")
    except:
        pass
    
    exit(1)

finally:
    cleanup()

print("\n" + "="*60)
print("🎉 DONE!")
print("="*60)