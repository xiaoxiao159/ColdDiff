  CUDA_VISIBLE_DEVICES=3 python fast_ddpm_main.py \
  --config twg_cdnet_nort.yml \
  --dataset LDFDCT \
  --exp LoDoPaB_BC\
  --doc LoDoPaB \
  --sample \
  --fid    --resume_training 

  CUDA_VISIBLE_DEVICES=4 python fast_ddpm_main.py \
  --config twg_cdnet.yml \
  --dataset LDFDCT \
  --exp LoDoPaB_BC\
  --doc LoDoPaB \
  --sample \
  --fid
  --resume_training