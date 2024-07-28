import argparse
import os
import sys

sys.path.append("../model")

from genMusic import GenMusic

os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "1"

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS


def genrateTheSong(ctx: str, lenOfOut: int, song_len: int):
    model = RWKV(model="./../model/L20-D512-x0601/rwkv-final.pth", strategy="cuda fp16")
    pipeline = PIPELINE(model, "./../model/src/tokenizer/tokenizer-midi/tokenizer.json")

    data = []

    args = PIPELINE_ARGS(
        temperature=1.5,
        top_p=0.9,
        top_k=100,  # top_k = 0 then ignore
        alpha_frequency=0.25,
        alpha_presence=0.25,
        alpha_decay=0.996,  # gradually decay the penalty
        token_ban=[],  # ban the generation of some tokens
        token_stop=["<end>"],  # stop generation whenever you see any token here
        chunk_len=256,
    )  # split input into chunks to save VRAM (shorter -> slower)

    while len(data) <= lenOfOut:
        pipeline.generate(ctx, args=args, callback=lambda token: data.append(token))
        ctx = " ".join(map(str, data[:5]))
        print(len(data))

    p1 = GenMusic(data[:lenOfOut])

    p1.mix_lines(song_len).export("./output.mp3", format="mp3")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", "--ctx", help="Enter the Context")
    arg_parser.add_argument(
        "-t", "--out_len", help="Enter the Lenth of uniqe output", default=100, type=int
    )
    arg_parser.add_argument(
        "-st", "--song_len", help="Enter the Lenth of song len", default=60, type=int
    )

    args = arg_parser.parse_args()
    genrateTheSong(args.ctx, args.out_len, args.song_len)
