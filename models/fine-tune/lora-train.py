"""
QLoRA fine-tuning for PoliceAI law checker.

Model:   nvidia/Llama-3_3-Nemotron-Super-49B-v1_5
         Chosen because:
           - NVIDIA's own reasoning-optimised model (post-trained for
             reasoning, RAG, tool-calling; derived from Llama-3.3-70B via
             Neural Architecture Search / "Puzzle" NAS)
           - 49B × NF4 ≈ 25 GB — fits with even more headroom than the 70B
             in the DGX Spark 128 GB unified memory pool
           - After fine-tuning: convert to TensorRT-LLM engine and serve
             locally on DGX Spark — replaces the cloud API call
           - Nemotron-4-340B cannot be fine-tuned on a single DGX Spark;
             even NF4 weights alone reach ~170 GB

         NOTE — this is a custom (DeciLM) architecture:
           - Requires trust_remote_code=True (set below)
           - NAS produces NON-UNIFORM layers: some attention blocks are
             removed/replaced and FFN widths vary. LoRA target-module name
             matching still works (PEFT only patches modules that exist),
             but it is not a clean uniform Llama stack.
           - To enable reasoning, the system prompt must contain
             "detailed thinking on" (see SYSTEM_PROMPT below).

Hardware: NVIDIA DGX Spark (GB10 Grace Blackwell, 128 GB unified memory)
Method:   QLoRA — NF4 4-bit quantised base + BF16 LoRA adapters (r=16)

Estimated memory during training
  Base weights (NF4):        ~25 GB
  LoRA adapter params:        ~0.3 GB
  Optimizer states (paged):   ~1 GB
  Activations (grad ckpt):   ~12 GB
  Total:                     ~38 GB  →  well within 128 GB

Usage:
  python train_lora.py \
    --data_path  ../../models/rag/training_data/legal_qa.jsonl \
    --output_dir ./checkpoints/nemotron-super-49b-policeai-v1 \
    --hf_token   hf_XXXX
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

# ---------------------------------------------------------------------------
# DGX Spark optimisations
# ---------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True   # Blackwell supports TF32
torch.backends.cudnn.allow_tf32 = True

MODEL_ID = "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5"

SYSTEM_PROMPT = (
    # "detailed thinking on" activates the Nemotron reasoning mode.
    "detailed thinking on\n"
    "You are PoliceAI, a real-time legal advisor for on-duty law enforcement officers. "
    "Your role is to provide legally-grounded, de-escalation-focused guidance during "
    "active incidents. You NEVER give an opinion on guilt or innocence. All advice must "
    "be phrased as guidance, not commands. Always cite specific case law or statutes. "
    "You must reason step by step through the legal situation before producing your final "
    "structured JSON response."
)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA fine-tune Nemotron-Super-49B for PoliceAI")
    p.add_argument("--model_id", default=MODEL_ID)
    p.add_argument("--data_path", default="../../models/rag/training_data/legal_qa.jsonl")
    p.add_argument("--output_dir", default="./checkpoints/nemotron-super-49b-policeai-v1")
    p.add_argument("--hf_token", default=os.getenv("HF_TOKEN", ""))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--merge_and_save", action="store_true",
                   help="Merge LoRA weights into base model and save a single model directory")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        sys.exit(f"No records found in {path}")

    print(f"Loaded {len(records)} training examples from {path}")
    return Dataset.from_list(records)


def format_chat(example: dict, tokenizer) -> dict:
    """Apply the tokeniser's chat template to a messages list."""
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# ---------------------------------------------------------------------------
# Quantisation config (NF4 — fits on single DGX Spark)
# ---------------------------------------------------------------------------

def bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,   # BF16 compute on Blackwell
        bnb_4bit_use_double_quant=True,           # double-quant saves ~0.4 GB extra
    )


# ---------------------------------------------------------------------------
# LoRA config
# ---------------------------------------------------------------------------

def lora_config(r: int, alpha: int, dropout: float) -> LoraConfig:
    # Target the attention + MLP projection matrices. Nemotron-Super is a NAS
    # (DeciLM) derivative of Llama, so layers are non-uniform: PEFT matches these
    # names only on the blocks where they actually exist (some attention blocks
    # are removed/replaced, and FFN widths vary). Adding gate/up/down projections
    # improves domain adaptation quality for specialised tasks like legal advice.
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        modules_to_save=["embed_tokens", "lm_head"],  # keep token embeddings trainable
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ---- tokeniser ----
    print(f"Loading tokeniser: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        token=args.hf_token or None,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # required for training (not left-pad)

    # ---- dataset ----
    raw_ds = load_dataset(args.data_path)
    ds = raw_ds.map(
        lambda ex: format_chat(ex, tokenizer),
        remove_columns=raw_ds.column_names,
    )

    # 90/10 train/eval split
    split = ds.train_test_split(test_size=0.1, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"Train: {len(train_ds)}  |  Eval: {len(eval_ds)}")

    # ---- model (4-bit) ----
    print(f"Loading model in NF4: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config(),
        device_map="auto",               # DGX Spark: single GPU, auto maps correctly
        trust_remote_code=True,
        token=args.hf_token or None,
        attn_implementation="flash_attention_2",  # FA2 on Blackwell; if the DeciLM
                                                   # custom code rejects it, drop this arg
    )
    model.config.use_cache = False           # required for gradient checkpointing
    model.config.pretraining_tp = 1

    # ---- prepare for QLoRA ----
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(model, lora_config(args.lora_r, args.lora_alpha, args.lora_dropout))
    model.print_trainable_parameters()

    # ---- training args ----
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",          # paged optimiser keeps optimizer states in CPU RAM
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        fp16=False,
        bf16=True,                         # BF16 on Blackwell is native
        max_grad_norm=0.3,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="none",                  # swap to "wandb" if tracking is configured
        seed=args.seed,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,                     # disable packing — scenes have variable lengths
    )

    # ---- trainer ----
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    print("Starting training …")
    trainer.train()

    print(f"Saving LoRA adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # ---- optional: merge adapters into base model ----
    if args.merge_and_save:
        _merge_and_save(args)


def _merge_and_save(args: argparse.Namespace):
    """
    Merges LoRA adapters back into the base model weights and saves a
    standalone model directory that can be converted to a TensorRT-LLM
    engine for low-latency serving on DGX Spark.

    Convert to TRT-LLM engine after this step:
      trtllm-build --checkpoint_dir <merged_dir> \
                   --output_dir    ./trt-engines/policeai-super-49b \
                   --gemm_plugin   bfloat16 \
                   --max_batch_size 4
    """
    from peft import AutoPeftModelForCausalLM

    merged_dir = args.output_dir + "-merged"
    print(f"Merging LoRA into base weights → {merged_dir}")

    merged_model = AutoPeftModelForCausalLM.from_pretrained(
        args.output_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    merged_model = merged_model.merge_and_unload()
    merged_model.save_pretrained(merged_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    tokenizer.save_pretrained(merged_dir)

    print(f"Merged model saved to {merged_dir}")
    print("Next step: convert to TensorRT-LLM engine for DGX Spark serving.")


if __name__ == "__main__":
    main()
