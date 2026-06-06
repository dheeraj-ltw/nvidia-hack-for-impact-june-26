python lora-train.py \
    --merge_only \
    --to_gguf \
    --adapter_path  ../checkpoints/nemotron-3-super \
    --llama_cpp_dir /home/nvidia/llama.cpp \
    --gguf_quant    Q4_K_M \
    --hf_token      $HF_TOKEN