
python quantize.py \
  --model-path /home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b \
  --clip-range weight_mse \
  --bits 4 --group-size 128 \
  --output-dir ./quantized_models/rtn_w4_weightmse


python quantize.py \
  --model-path /home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b \
  --clip-range linear_response \
  --bits 4 --group-size 128 \
  --output-dir ./quantized_models/rtn_w4_linear


python quantize.py \
  --model-path /home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b \
  --clip-range mixed --inner linear --lam 0.5 \
  --bits 4 --group-size 128 --n-calib 128 \
  --output-dir ./quantized_models/rtn_w4_mixed
  
python quantize.py \
  --model-path /home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b \
  --clip-range sigma_aware --lam 0.9 \
  --bits 4 --group-size 128 --n-calib 128 \
  --output-dir ./quantized_models/rtn_w4_sigma


Llama 3.1. /home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
Mistral 7B /home/DATA/prometheus/anh/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1 
Qwen2.5 /home/DATA/prometheus/anh/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796 
