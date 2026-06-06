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
Method:   QLoRA — FP4 4-bit quantised base + BF16 LoRA adapters (r=16)
          FP4 is used (instead of NF4) because GB10 Blackwell has native
          FP4 tensor cores (5th-gen), so it is the hardware-aligned 4-bit
          format on DGX Spark. Override with --quant_type nf4 if desired.

Environment (DGX Spark specifics — IMPORTANT):
  - GB10 is compute capability sm_121 and requires CUDA 13.0 + PyTorch >= 2.9.
    Install the aarch64 cu130 wheel, NOT the cu124 build:
        pip install torch --index-url https://download.pytorch.org/whl/cu130
    (An older torch raises:
       AttributeError: module 'torch' has no attribute 'float8_e8m0fnu'
     when transformers imports its FP8 integration.)
  - The DeciLM custom code only implements "flash_attention_2" and "eager"
    (NO SDPA path). flash-attn has no working sm_121 (GB10) wheel, so we fall
    back to attn_implementation="eager". Do NOT use "sdpa" or "flash_attention_2".
  - The custom DeciLM remote code for this model is written against the
    transformers 4.44-4.48 API and imports NEED_SETUP_CACHE_CLASSES_MAPPING,
    which was REMOVED in transformers >= ~4.50. You MUST pin transformers
    to 4.48.3 (NVIDIA's documented recommendation), otherwise loading fails:
       ImportError: cannot import name 'NEED_SETUP_CACHE_CLASSES_MAPPING'
                    from 'transformers.generation.utils'

Estimated memory during training
  Base weights (FP4):        ~25 GB
  LoRA adapter params:        ~0.3 GB
  Optimizer states (paged):   ~1 GB
  Activations (grad ckpt):   ~12 GB
  Total:                     ~38 GB  →  well within 128 GB

Usage:
  # Train on train.jsonl, evaluating against val.jsonl during training:
  python lora-train.py \
    --data_path     ../data/train.jsonl \
    --val_data_path ../data/val.jsonl \
    --output_dir    ./checkpoints/nemotron-super-49b-policeai-v1 \
    --hf_token      hf_XXXX

  # Later, re-evaluate the fine-tuned adapter against val.jsonl to get a score
  # (eval loss + perplexity) WITHOUT retraining:
  python lora-train.py \
    --eval_only \
    --adapter_path  ./checkpoints/nemotron-super-49b-policeai-v1 \
    --val_data_path ../data/val.jsonl \
    --hf_token      hf_XXXX

  # Merge LoRA -> bf16 and convert to a GGUF that llama.cpp can serve
  # (--merge_only skips training and operates on the saved checkpoint):
  python lora-train.py \
    --merge_only \
    --to_gguf \
    --adapter_path  ./checkpoints/nemotron-super-49b-policeai-v1 \
    --llama_cpp_dir /path/to/llama.cpp \
    --gguf_quant    Q4_K_M \
    --hf_token      hf_XXXX
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
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
    p.add_argument("--revision", default="main",
                   help="Pin the HF model revision (commit hash/tag). Pinning a "
                        "commit stops the custom DeciLM remote code from silently "
                        "re-downloading on each run.")
    p.add_argument("--data_path", default="../data/train.jsonl",
                   help="JSONL training set (one {\"messages\": [...]} object per line).")
    p.add_argument("--val_data_path", default="../data/val.jsonl",
                   help="JSONL evaluation set. Used as the eval split during training "
                        "and as the scoring set in --eval_only mode. If the file does "
                        "not exist, training falls back to a 90/10 split of --data_path.")
    p.add_argument("--output_dir", default="./checkpoints/nemotron-super-49b-policeai-v1")
    p.add_argument("--hf_token", default=os.getenv("HF_TOKEN", ""))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--quant_type", default="fp4", choices=["fp4", "nf4"],
                   help="4-bit quant format. fp4 aligns with GB10 Blackwell's "
                        "native FP4 tensor cores; nf4 is the classic QLoRA format.")
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--merge_and_save", action="store_true",
                   help="Merge LoRA weights into base model and save a single bf16 model directory")
    p.add_argument("--merge_only", action="store_true",
                   help="Skip training. Merge an existing adapter (from --adapter_path "
                        "or --output_dir) and optionally convert to GGUF (--to_gguf).")
    p.add_argument("--to_gguf", action="store_true",
                   help="After merging, convert the bf16 model to GGUF for llama.cpp. "
                        "Requires --llama_cpp_dir. Implies --merge_and_save.")
    p.add_argument("--llama_cpp_dir", default="",
                   help="Path to a local llama.cpp checkout (must contain "
                        "convert_hf_to_gguf.py). Used by --to_gguf.")
    p.add_argument("--gguf_quant", default="Q4_K_M",
                   help="GGUF quantisation type to produce after the f16 conversion "
                        "(e.g. Q4_K_M, Q5_K_M, Q8_0). Set to 'none' to keep f16 only.")
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training. Load the fine-tuned LoRA adapter from "
                        "--adapter_path and evaluate it against --val_data_path, "
                        "reporting eval loss and perplexity.")
    p.add_argument("--adapter_path", default="",
                   help="Path to a saved LoRA adapter dir for --eval_only. "
                        "Defaults to --output_dir when empty.")
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

    print(f"Loaded {len(records)} examples from {path}")
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


def prepare_dataset(path: str, tokenizer) -> Dataset:
    """Load a JSONL chat dataset and render it to a single `text` column."""
    raw_ds = load_dataset(path)
    return raw_ds.map(
        lambda ex: format_chat(ex, tokenizer),
        remove_columns=raw_ds.column_names,
    )


# ---------------------------------------------------------------------------
# Quantisation config (4-bit — fits on single DGX Spark)
# ---------------------------------------------------------------------------

def bnb_config(quant_type: str = "fp4") -> BitsAndBytesConfig:
    # FP4 maps onto GB10 Blackwell's native 4-bit tensor cores; NF4 is the
    # classic QLoRA format (often marginally higher quality, no native HW path).
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
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
# Loaders (shared by training and eval-only)
# ---------------------------------------------------------------------------

def load_tokenizer(source: str, revision: str, hf_token: str):
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        revision=revision,
        token=hf_token or None,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # required for training (not left-pad)
    return tokenizer


def load_base_model(args: argparse.Namespace):
    """Load the NF4/FP4-quantised base model, pinned to the single DGX Spark GPU."""
    print(f"Loading model in {args.quant_type.upper()}: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        quantization_config=bnb_config(args.quant_type),
        device_map={"": 0},              # DGX Spark is a SINGLE GPU. "auto" lets
                                         # accelerate offload some layers to CPU/disk,
                                         # which bnb 4-bit rejects. The ~25 GB FP4
                                         # model fits easily in the 128 GB unified
                                         # pool, so pin everything to cuda:0.
        trust_remote_code=True,
        token=args.hf_token or None,
        attn_implementation="eager",     # DeciLM custom code supports only FA2 or
                                         # eager (no SDPA path); FA2 has no sm_121
                                         # (GB10) wheel, so eager is the only option
    )
    model.config.pretraining_tp = 1
    return model


def report_metrics(metrics: dict) -> dict:
    """Print eval metrics and derive perplexity from eval loss."""
    eval_loss = metrics.get("eval_loss")
    if eval_loss is not None:
        try:
            metrics["eval_perplexity"] = math.exp(eval_loss)
        except OverflowError:
            metrics["eval_perplexity"] = float("inf")

    print("\n================ Evaluation results ================")
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float):
            print(f"  {key:24s} {value:.4f}")
        else:
            print(f"  {key:24s} {value}")
    print("====================================================\n")
    return metrics


# ---------------------------------------------------------------------------
# Eval-only mode
# ---------------------------------------------------------------------------

def run_eval(args: argparse.Namespace):
    """Load a fine-tuned LoRA adapter and score it against --val_data_path."""
    adapter_path = args.adapter_path or args.output_dir
    if not Path(adapter_path).exists():
        sys.exit(f"--eval_only: adapter path not found: {adapter_path}")
    if not Path(args.val_data_path).exists():
        sys.exit(f"--eval_only: validation file not found: {args.val_data_path}")

    # Tokeniser was saved alongside the adapter during training; prefer it.
    print(f"Loading tokeniser from adapter dir: {adapter_path}")
    tokenizer = load_tokenizer(adapter_path, args.revision, args.hf_token)

    eval_ds = prepare_dataset(args.val_data_path, tokenizer)
    print(f"Eval examples: {len(eval_ds)}")

    base_model = load_base_model(args)
    base_model.config.use_cache = True  # eval/inference benefits from KV cache

    print(f"Attaching LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    eval_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.per_device_batch_size,
        bf16=True,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=eval_args,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    print("Evaluating fine-tuned adapter on validation set …")
    metrics = report_metrics(trainer.evaluate())

    metrics_path = Path(adapter_path) / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote eval metrics to {metrics_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.eval_only:
        run_eval(args)
        return

    if args.merge_only:
        merged_dir = _merge_and_save(args)
        if args.to_gguf:
            _convert_to_gguf(merged_dir, args)
        return

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ---- tokeniser ----
    print(f"Loading tokeniser: {args.model_id}")
    tokenizer = load_tokenizer(args.model_id, args.revision, args.hf_token)

    # ---- datasets ----
    train_ds = prepare_dataset(args.data_path, tokenizer)
    if args.val_data_path and Path(args.val_data_path).exists():
        # Dedicated held-out validation set (preferred).
        eval_ds = prepare_dataset(args.val_data_path, tokenizer)
    else:
        # Fallback: carve a 90/10 eval split out of the training data.
        print(f"No val file at {args.val_data_path!r}; using a 90/10 split of train.")
        split = train_ds.train_test_split(test_size=0.1, seed=args.seed)
        train_ds, eval_ds = split["train"], split["test"]
    print(f"Train: {len(train_ds)}  |  Eval: {len(eval_ds)}")

    # ---- model (4-bit) ----
    model = load_base_model(args)
    model.config.use_cache = False           # required for gradient checkpointing

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

    # ---- final evaluation on the held-out validation set ----
    metrics = report_metrics(trainer.evaluate())
    metrics_path = Path(args.output_dir) / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote eval metrics to {metrics_path}")

    # ---- optional: merge adapters into base model ----
    if args.merge_and_save or args.to_gguf:
        merged_dir = _merge_and_save(args)
        if args.to_gguf:
            _convert_to_gguf(merged_dir, args)


def _merge_and_save(args: argparse.Namespace):
    """
    Merges LoRA adapters back into the base model weights and saves a
    standalone, UN-QUANTISED (bf16) HF model directory. This bf16 directory is
    the artifact required by BOTH downstream serving paths:

    A) llama.cpp / GGUF (recommended on DGX Spark — see compatibility notes):
       The DeciLM/Nemotron-Super architecture is supported by llama.cpp's
       converter. NOTE: the converter cannot read 4-bit (bnb/modelopt) weights,
       so you MUST convert from this merged bf16 dir, not from the adapter.

         # one-time: build llama.cpp from source for GB10 (sm_121, CUDA 13)
         git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
         cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121
         cmake --build build --config Release -j

         # convert merged HF model -> GGUF, then quantise to Q4_K_M
         python convert_hf_to_gguf.py <merged_dir> \
                --outfile policeai-super-49b-f16.gguf --outtype f16
         ./build/bin/llama-quantize policeai-super-49b-f16.gguf \
                policeai-super-49b-Q4_K_M.gguf Q4_K_M

         # serve / run inference
         ./build/bin/llama-server -m policeai-super-49b-Q4_K_M.gguf --n-gpu-layers 999

    B) TensorRT-LLM engine (lowest latency, more build effort):
         trtllm-build --checkpoint_dir <merged_dir> \
                      --output_dir    ./trt-engines/policeai-super-49b \
                      --gemm_plugin   bfloat16 \
                      --max_batch_size 4
    """
    from peft import AutoPeftModelForCausalLM

    adapter_path = args.adapter_path or args.output_dir
    if not Path(adapter_path).exists():
        sys.exit(f"Adapter path not found: {adapter_path}")

    merged_dir = adapter_path.rstrip("/") + "-merged"
    print(f"Merging LoRA ({adapter_path}) into base weights → {merged_dir}")

    # Merge on CPU: the bf16 49B base is ~98 GB. Loading it on the GPU via
    # device_map="auto" risks OOM/offload; merging in the unified host RAM is
    # safe and the result is identical. trust_remote_code is REQUIRED because
    # the base is a custom DeciLM architecture.
    merged_model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        token=args.hf_token or None,
    )
    merged_model = merged_model.merge_and_unload()
    merged_model.save_pretrained(merged_dir, safe_serialization=True)

    tokenizer = load_tokenizer(adapter_path, args.revision, args.hf_token)
    tokenizer.save_pretrained(merged_dir)

    print(f"Merged bf16 model saved to {merged_dir}")
    print("Next step: convert to GGUF for llama.cpp (see _merge_and_save docstring), "
          "or build a TensorRT-LLM engine.")
    return merged_dir


def _convert_to_gguf(merged_dir: str, args: argparse.Namespace):
    """
    Convert the merged bf16 HF model to GGUF and (optionally) quantise it so it
    can be loaded by llama.cpp / llama-cpp-python for inference.

    The DeciLM/Nemotron-Super architecture is supported by llama.cpp's
    convert_hf_to_gguf.py, but ONLY from un-quantised (bf16/f16) weights — which
    is exactly what _merge_and_save produced.
    """
    if not args.llama_cpp_dir:
        sys.exit("--to_gguf requires --llama_cpp_dir pointing to a llama.cpp checkout.")

    converter = Path(args.llama_cpp_dir) / "convert_hf_to_gguf.py"
    if not converter.exists():
        sys.exit(f"convert_hf_to_gguf.py not found at {converter}. "
                 f"Clone https://github.com/ggml-org/llama.cpp first.")

    out_name = Path(merged_dir).name
    f16_path = str(Path(merged_dir).parent / f"{out_name}-f16.gguf")

    print(f"Converting {merged_dir} → {f16_path}")
    subprocess.run(
        [sys.executable, str(converter), merged_dir,
         "--outfile", f16_path, "--outtype", "f16"],
        check=True,
    )
    print(f"GGUF (f16) written to {f16_path}")

    if args.gguf_quant and args.gguf_quant.lower() != "none":
        # llama-quantize lives in the build dir after compiling llama.cpp.
        candidates = [
            Path(args.llama_cpp_dir) / "build" / "bin" / "llama-quantize",
            Path(args.llama_cpp_dir) / "llama-quantize",
        ]
        quant_bin = next((c for c in candidates if c.exists()), None)
        if quant_bin is None:
            print("WARNING: llama-quantize binary not found; keeping f16 GGUF only. "
                  "Build llama.cpp (cmake --build) to enable quantisation.")
            return

        quant_path = str(Path(merged_dir).parent / f"{out_name}-{args.gguf_quant}.gguf")
        print(f"Quantising → {quant_path} ({args.gguf_quant})")
        subprocess.run([str(quant_bin), f16_path, quant_path, args.gguf_quant], check=True)
        print(f"Quantised GGUF written to {quant_path}")
        print(f"Run it with: {Path(args.llama_cpp_dir) / 'build' / 'bin' / 'llama-server'} "
              f"-m {quant_path} --n-gpu-layers 999")


if __name__ == "__main__":
    main()
