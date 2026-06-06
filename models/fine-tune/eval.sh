python lora-train.py \
    --eval_only \
    --adapter_path  ../checkpoints/nemotron-3-super \
    --val_data_path ../data/val.jsonl \
    --hf_token      $HF_TOKEN