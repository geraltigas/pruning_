CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --input_path data/xsum.jsonl \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --enable_qh2o_cache \
    --output_path output/xsum_llama_2_7b_chat_h2o_q.jsonl
