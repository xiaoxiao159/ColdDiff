  CUDA_VISIBLE_DEVICES=4 python fast_ddpm_main.py \
  --config ldfd_cold.yml \
  --dataset LDFDCT \
  --exp LoDoPaB_10_stage2_consine \
  --doc LoDoPaB \
  --resume_training
  --sample \
  --fid  --resume_training 