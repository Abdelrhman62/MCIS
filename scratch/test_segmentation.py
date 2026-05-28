import pandas as pd
from transformers import AutoTokenizer

# 1. Load a single sample from the Baheya dataset
df = pd.read_parquet("data/frozen/m1_model_ready/m1_model_ready.parquet")
sample_text = df.iloc[5]["text_section_tagged"]  # Pick a random row
print(f"--- ORIGINAL REPORT TEXT (Length: {len(sample_text.split())} words) ---\n")
print(sample_text)
print("\n" + "="*80 + "\n")

# 2. Load the exact tokenizer used in MCIS_Best
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")

# 3. Apply the exact segmentation logic from src/models/encoder.py
encoded = tokenizer(
    sample_text,
    max_length=128,
    truncation=True,
    return_overflowing_tokens=True,
    stride=0,
    padding="max_length",
    return_tensors="pt",
)

input_ids = encoded["input_ids"]
n_segments = input_ids.shape[0]

print(f"--- HOW THE MODEL SEES IT ({n_segments} SEGMENTS OF 128 TOKENS) ---\n")

for i in range(n_segments):
    # Decode the tokens back to human-readable text
    # skip_special_tokens=False so we can see the [CLS] and [SEP] boundaries!
    decoded_chunk = tokenizer.decode(input_ids[i], skip_special_tokens=False)
    
    print(f"📌 SEGMENT {i+1}:")
    print(decoded_chunk)
    print("-" * 80)
