
vllm serve /home/cjy/pretrained_models/qwen3.8-27b \
  --served-model-name qwen3.5-9b \
  --trust-remote-code \
  --max-model-len 32768 \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --max-num-seqs 4
