  CUDA_VISIBLE_DEVICES=0 python fast_ddpm_main.py \
  --config ldfd_cold.yml \
  --dataset LDFDCT \
  --exp LoDoPaB_10 \
  --doc LoDoPaB \
  --sample \
  --fid  --resume_training 