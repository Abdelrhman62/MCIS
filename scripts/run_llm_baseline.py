"""LLM Baseline Evaluation.

Evaluates an LLM (OpenAI GPT-4o or Anthropic Claude 3.5) on the held-out test set
using zero-shot prompting to predict the 10 target axes.

Usage:
    export OPENAI_API_KEY="sk-..."
    python scripts/run_llm_baseline.py --provider openai --model gpt-4o --config configs/LLM_base.yaml

    export ANTHROPIC_API_KEY="sk-..."
    python scripts/run_llm_baseline.py --provider anthropic --model claude-3-5-sonnet-20240620 --config configs/LLM_base.yaml
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.label_vocab import LabelVocab
from src.utils.config import BenchmarkConfig
from src.utils.logging import setup_console_logger
from scripts.run_full_evaluation import (
    compute_singlepick_metrics,
    compute_multilabel_metrics,
    compute_rare_common_f1,
    format_report
)

log = setup_console_logger("mcis.llm_baseline")

def build_prompt(text: str, vocab: LabelVocab, cfg: BenchmarkConfig, prompt_type: str) -> str:
    """Constructs the prompt for the LLM."""
    prompt = (
        "You are an expert breast cancer oncology coder. Your task is to extract "
        "ICD-O-3 and ICD-11 oncology codes from the following pathology report.\n\n"
        "REPORT:\n"
        f"\"\"\"{text}\"\"\"\n\n"
        "INSTRUCTIONS:\n"
    )

    if prompt_type == "structured":
        prompt += "Extract the exact codes for the following 10 axes. You must ONLY use codes from the 'Valid Codes' lists provided below.\n"
        for axis in cfg.all_axes:
            valid_codes = list(vocab[axis].code_to_idx.keys())
            if axis in cfg.axis_types.single_pick:
                prompt += f"- {axis}: (Single-pick. Choose exactly ONE code from: {valid_codes})\n"
            else:
                prompt += f"- {axis}: (Multi-label. Choose ZERO OR MORE codes from: {valid_codes})\n"
    else:
        # Pure Zero-Shot
        prompt += "Extract the correct oncology codes for the following 10 target axes based on standard ICD-O-3 and ICD-11 conventions.\n"
        for axis in cfg.all_axes:
            if axis in cfg.axis_types.single_pick:
                prompt += f"- {axis}: (Single-pick. Return exactly ONE standard code)\n"
            else:
                prompt += f"- {axis}: (Multi-label. Return a list of standard codes)\n"

    prompt += (
        "\nReturn ONLY a valid JSON object matching this exact format, with no markdown formatting or other text:\n"
        "{\n"
    )
    for i, axis in enumerate(cfg.all_axes):
        comma = "," if i < len(cfg.all_axes) - 1 else ""
        if axis in cfg.axis_types.single_pick:
            prompt += f'  "{axis}": "CODE"{comma}\n'
        else:
            prompt += f'  "{axis}": ["CODE1", "CODE2"]{comma}\n'
    prompt += "}\n"
    return prompt

def call_openai(prompt: str, model: str) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def call_anthropic(prompt: str, model: str) -> dict:
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.content[0].text
    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)

def call_local_hf(prompt: str, pipeline) -> dict:
    import re
    messages = [
        {"role": "system", "content": "You are a precise JSON-generating system. You only output valid JSON dictionaries."},
        {"role": "user", "content": prompt}
    ]
    outputs = pipeline(messages, max_new_tokens=1024, temperature=0.1, do_sample=False)
    text = outputs[0]["generated_text"][-1]["content"]
    
    # Try to extract JSON using regex if the model output conversational padding
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        text = json_match.group(0)
    
    return json.loads(text)

def main():
    parser = argparse.add_argument_group("LLM Baseline")
    parser.add_argument("--provider", choices=["openai", "anthropic", "local"], required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt-type", choices=["zero-shot", "structured"], default="structured", help="Essen 2025 prompt strategy")
    parser.add_argument("--config", type=Path, default=Path("configs/LLM_base.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/llm_baseline_test.md"))
    parser.add_argument("--cache", type=Path, default=Path("results/llm_predictions_cache.json"))
    args = parser.parse_args()

    cfg = BenchmarkConfig.from_yaml(args.config)
    vocab = LabelVocab.load_json(cfg.data.label_vocab)
    df = pd.read_parquet(cfg.data.parquet)
    test_df = df[df["split"] == cfg.data.test_value].copy()
    
    log.info(f"Evaluating {len(test_df)} test records using {args.provider} ({args.model})")

    # Load cache if exists
    predictions = {}
    if args.cache.exists():
        with open(args.cache, "r") as f:
            predictions = json.load(f)
        log.info(f"Loaded {len(predictions)} cached predictions.")

    # Load HF Pipeline if local
    hf_pipeline = None
    if args.provider == "local":
        import torch
        from transformers import pipeline
        log.info(f"Loading local model {args.model} into VRAM...")
        hf_pipeline = pipeline("text-generation", model=args.model, torch_dtype=torch.float16, device_map="auto")
        log.info("Model loaded successfully.")

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        pid = str(row["patient_id"])
        if pid in predictions:
            continue
            
        text = row[cfg.data.text_field]
        prompt = build_prompt(text, vocab, cfg, args.prompt_type)
        
        success = False
        retries = 0
        while not success and retries < 3:
            try:
                if args.provider == "openai":
                    pred_json = call_openai(prompt, args.model)
                elif args.provider == "anthropic":
                    pred_json = call_anthropic(prompt, args.model)
                elif args.provider == "local":
                    pred_json = call_local_hf(prompt, hf_pipeline)
                predictions[pid] = pred_json
                success = True
                
                # Save cache progressively
                with open(args.cache, "w") as f:
                    json.dump(predictions, f, indent=2)
                    
            except Exception as e:
                retries += 1
                log.warning(f"Error on {pid}: {e}. Retrying ({retries}/3)...")
                time.sleep(2)

    # Convert predictions to pseudo-logits for metrics
    import torch
    logits_dict = {ax: [] for ax in cfg.all_axes}
    targets_dict = {ax: [] for ax in cfg.all_axes}
    
    for idx, row in test_df.iterrows():
        pid = str(row["patient_id"])
        pred = predictions.get(pid, {})
        
        for axis in cfg.all_axes:
            K = vocab[axis].num_classes
            c2i = vocab[axis].code_to_idx
            
            # Target
            true_idx = row[f"label_{axis}"]
            targets_dict[axis].append(true_idx)
            
            # Prediction
            pred_logits = np.zeros(K)
            pred_val = pred.get(axis)
            
            if axis in cfg.axis_types.single_pick:
                if isinstance(pred_val, str) and pred_val in c2i:
                    pred_logits[c2i[pred_val]] = 10.0 # High confident logit
            else:
                if isinstance(pred_val, list):
                    for v in pred_val:
                        if isinstance(v, str) and v in c2i:
                            pred_logits[c2i[v]] = 10.0
                            
            logits_dict[axis].append(pred_logits)

    # Convert to tensors
    for axis in cfg.all_axes:
        logits_dict[axis] = torch.tensor(np.array(logits_dict[axis]))
        targets_dict[axis] = torch.tensor(np.array(targets_dict[axis]))

    # Compute metrics
    per_axis = {}
    rare_common = {}
    rare_threshold = cfg.eval.rare_code_threshold
    
    for axis in cfg.all_axes:
        lg = logits_dict[axis]
        tg = targets_dict[axis]
        K = vocab[axis].num_classes
        
        if axis in cfg.axis_types.single_pick:
            per_axis[axis] = compute_singlepick_metrics(lg, tg, K)
        else:
            per_axis[axis] = compute_multilabel_metrics(lg, tg)
            
        rare_codes = vocab[axis].rare_codes(rare_threshold)
        rare_idx = {vocab[axis].code_to_idx[c] for c in rare_codes if c in vocab[axis].code_to_idx}
        rare_common[axis] = compute_rare_common_f1(lg, tg, K, rare_idx)

    # Format report (empty ranking/ece/kappa for LLM)
    report = format_report(per_axis, {}, rare_common, [], {}, None, args, len(test_df))
    args.output.write_text(report)
    log.info(f"Report written: {args.output}")

if __name__ == "__main__":
    main()
