#!/usr/bin/env python3
"""
Test your fine-tuned Qwen3-VL model
"""

import torch
from unsloth import FastVisionModel
from PIL import Image
import os

print("\n" + "="*60)
print("TESTING FINE-TUNED QWEN3-VL MODEL")
print("="*60)

# Load fine-tuned model
print("\n📦 Loading fine-tuned model...")
model, processor = FastVisionModel.from_pretrained(
    "./qwen3vl-xray-final",
    load_in_4bit=True,
)

# Enable inference mode
FastVisionModel.for_inference(model)
print("✅ Model loaded")

# Get a test image from your validation set
dataset_path = "./xr_bones_qwen_format/hf_dataset_fixed"
from datasets import DatasetDict
dataset = DatasetDict.load_from_disk(dataset_path)

print("\n🔍 Testing on validation samples...")
print("="*60)

# Test on 3 different samples
for i in range(3):
    sample = dataset['validation'][i]
    
    print(f"\n--- Test {i+1} ---")
    
    # Load image
    img_path = sample['image_path']
    if not os.path.exists(img_path):
        print(f"⚠️  Image not found: {img_path}")
        continue
    
    image = Image.open(img_path).convert('RGB')
    
    # Show ground truth
    print(f"Image: {os.path.basename(img_path)}")
    print(f"Ground truth: {sample['target_output'][:100]}...")
    
    # Prepare input
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Analyze this X-ray image and identify any bone fractures."}
        ]
    }]
    
    # Generate response
    # Step 1: Get text with image tokens
    text = processor.apply_chat_template(
        messages,
        tokenize=False,  # Get text first
        add_generation_prompt=True
    )

    # Step 2: Process to get proper dict
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"  # Now returns proper dict
    ).to("cuda")

    # Step 3: Generate
    outputs = model.generate(**inputs)  # Works!
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    # Decode response
    response = processor.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    
    print(f"Model prediction: {response}")
    print("-" * 60)

print("\n✅ Testing complete!")

# Interactive testing
print("\n" + "="*60)
print("INTERACTIVE TESTING")
print("="*60)
print("\nYou can now test with your own images!")
print("Usage:")
print("""
from unsloth import FastVisionModel
from PIL import Image
import torch

model, processor = FastVisionModel.from_pretrained(
    "./qwen3vl-xray-final",
    load_in_4bit=True,
)
FastVisionModel.for_inference(model)

image = Image.open("your_xray.jpg")
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Analyze this X-ray for fractures."}
    ]
}]

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to("cuda")

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=128)
    
response = processor.decode(outputs[0], skip_special_tokens=True)
print(response)
""")