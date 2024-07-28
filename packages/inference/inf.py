from fileinput import filename
import os

import sys
import sys
sys.path.append('../model')

from genMusic import GenMusic

from src.tokenizer.midi_util import VocabConfig, convert_str_to_midi
# set these before import RWKV
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries


# from rwkv.model import RWKV
# from rwkv.utils import PIPELINE, PIPELINE_ARGS

# # download models: https://huggingface.co/BlinkDL
# model = RWKV(model='./../model/L20-D512-x0601/rwkv-8.pth', strategy='cuda fp16')
# pipeline = PIPELINE(model, "./../model/src/tokenizer/tokenizer-midi/tokenizer.json") # 220B_tokenizer0B_tokenizer.json is in https://github.com/BlinkDL/ChatRWKV
# # use pipeline = PIPELINE(model, "rwkv_vocab_v20230424") for rwkv "world" models

# ctx = "31:b 34:b 38:b t1 3d:b t5 3f:b 44:b t125 t56 49:b t31 47:b 49:0 t24 31:0 34:0 38:0 t2 3d:0 t5 31:b 33:b 3f:0 44:0 46:b 47:0 t7 36:b 3a:b t112 31:0 33:0 t7 33:b 36:0 3a:0 3f:b t7 36:b"

# data = []

# def my_print(s):
#     data.append(s)


# args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.7, top_k = 100, # top_k = 0 then ignore
#                      alpha_frequency = 0.25,
#                      alpha_presence = 0.25,
#                      alpha_decay = 0.996, # gradually decay the penalty
#                      token_ban = [], # ban the generation of some tokens
#                      token_stop = [], # stop generation whenever you see any token here
#                      chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)

# pipeline.generate(ctx, args=args, callback=my_print)

# print(data)
p1 = GenMusic(data=['3f:0', '48:0', 't2', '3d:b', 't85', '3d:0', '3d:0', '3f:b', 't7', '38:b', '40:b', '42:0', 't7', '37:b', '3f:0', 't56', '37:0', '3f:a', 't1', '37:a', 't15', '36:b', '37:0', '3f:0', 't1', '3b:b', 't56', '36:0', '38:a', '3b:0', 't1', '36:a', 't15', '38:0', '38:c', '38:0', 't1', '38:c', 't110', '38:0', '38:0', '38:a', 't1', '38:a', 't15', '38:0', '38:0', '38:a', 't47',
              '38:0', '3b:a', 't15', '3b:0', '3d:a', 't47', '3d:0', '3f:c', 't1', '37:c', 't31', '32:b', '33:0', '37:0', 't109', '32:0', '36:a', 't15', '36:0', '37:a', 't31', '37:0', '39:a', 't31', '39:0', '3b:c', 't1', '3e:c', 't37', '3b:0', '3e:0', '3e:a', 't2', '37:a', 't18', '37:0', '3e:0', '3e:a', 't2', '37:a', 't18', '37:0', '3e:0', '3e:b', 't1', '2b:b', 't37', '2b:0', '3e:0', '3e:a', 't2'])

p1.mix_lines().export("./output.mp3", format="mp3")

# out, state = model.forward([187, 510, 1563, 310, 247], None)
# print(out.detach().cpu().numpy())                   # get logits
# out, state = model.forward([187, 510], None)
# out, state = model.forward([1563], state)           # RNN has state (use deepcopy to clone states)
# out, state = model.forward([310, 247], state)
# print(out.detach().cpu().numpy())                   # same result as above
print('\n')
