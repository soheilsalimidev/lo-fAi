#!/bin/bash
cd ./RWKV-LM;
MODEL_TYPE="x060" # x060 => rwkv-6.0e
N_LAYER="20"
N_EMBD="512"
CTX_LEN="512" # !!! change magic_prime if you change ctx_len !!!
PROJ_DIR="/mnt/c/Users/sohei/OneDrive/Desktop/AItuneCraft/packages/model/drumL"$N_LAYER"-D"$N_EMBD"-"$MODEL_TYPE"1" # set
M_BSZ="24" 
LR_INIT="6e-4"
LR_FINAL="6e-5"
GRAD_CP=1 # 1 => slower, save VRAM; 0 => faster, more VRAM
EPOCH_SAVE=10 # save every 10 "miniepochs" (1 miniepoch = 40320 * ctx_len tokens) => decrease if your GPU is weak
N_NODE=1
GPU_PER_NODE=1
DS_BUCKET_MB=2
VOCAB_SIZE=2176

python train.py --wandb "aitune" --proj_dir $PROJ_DIR --my_testing $MODEL_TYPE \
 --ctx_len $CTX_LEN --my_pile_stage 3 --epoch_count 999999 --epoch_begin 0 \
 --data_file "/mnt/c/Users/sohei/OneDrive/Desktop/AItuneCraft/packages/model/src/midiData/drum_text_document" --my_exit_tokens 34234278 --magic_prime 66851 \
 --num_nodes $N_NODE --micro_bsz $M_BSZ --n_layer $N_LAYER --n_embd $N_EMBD --pre_ffn 0 --head_qk 0 \
 --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --my_pile_edecay 0 --data_type "binidx" --vocab_size $VOCAB_SIZE \
 --weight_decay 0.001 --epoch_save $EPOCH_SAVE --head_size_a 64 \
 --accelerator gpu --devices $GPU_PER_NODE --precision bf16 --strategy deepspeed_stage_2 --grad_cp $GRAD_CP --enable_progress_bar True --ds_bucket_mb $DS_BUCKET_MB

