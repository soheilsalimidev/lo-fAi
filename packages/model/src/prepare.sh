#!/bin/bash
cd ./RWKV-LM
MODEL_TYPE="x060"
N_LAYER="20"
N_EMBD="512"

CTX_LEN="512" # !!! change magic_prime if you change ctx_len !!!
PROJ_DIR="/mnt/c/Users/sohei/OneDrive/Desktop/AItuneCraft/packages/model/drumL"$N_LAYER"-D"$N_EMBD"-"$MODEL_TYPE"1"
VOCAB_SIZE=2176

python train.py --wandb "" --proj_dir $PROJ_DIR \
 --data_file "/mnt/c/Users/sohei/OneDrive/Desktop/AItuneCraft/packages/model/src/midiData/drum_text_document"  --data_type "binidx" --vocab_size $VOCAB_SIZE --my_testing $MODEL_TYPE \
 --ctx_len $CTX_LEN --my_pile_stage 1 --epoch_count 1 --epoch_begin 0 \
 --epoch_save 1 --weight_decay 0 --head_size_a 64 \
 --num_nodes 1 --micro_bsz 1 --n_layer $N_LAYER --n_embd $N_EMBD --pre_ffn 0 --head_qk 0 --my_exit_tokens 34234278 --magic_prime 66851\
 --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --my_pile_edecay 0 \
 --accelerator cpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 1

