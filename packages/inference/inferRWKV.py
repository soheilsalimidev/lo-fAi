import sys
sys.path.append("./../model")
sys.path.append("./../model/src/tokenizer")

from src.tokenizer.midi_to_str import convert_midi_bytes_to_str
from genMusic import GenMusic
import os
import argparse
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
import random



def relpath(p): return os.path.normpath(
    os.path.join(os.path.dirname(__file__), p))


os.environ["RWKV_JIT_ON"] = "0"
os.environ["RWKV_CUDA_ON"] = "0"


def genrateTheSong(ctx: str, lenOfOut: int, modelPath: str, temperature: float):
    model = RWKV(model=modelPath, strategy="cpu fp32")
    pipeline = PIPELINE(
        model, "./../model/src/tokenizer/tokenizer-midi/tokenizer.json")

    data = []

    args = PIPELINE_ARGS(
        temperature,
        top_p=0.8,
        top_k=8,
        # alpha_frequency=0.25,
        # alpha_presence=0.25,
        alpha_decay=0.997,  # gradually decay the penalty
        token_stop=["<end>"],
        chunk_len=512,
    )

    while len(data) <= lenOfOut:
        pipeline.generate(
            ctx, args=args, callback=lambda token: data.append(token))
        ctx = " ".join(map(str, data[:5]))
        print(len(data))

    return data


def mixTheSongs(song_len, ctx, out_len, te=1.0):
    pinio, drum = genrateTheSong(ctx, out_len,
                                 relpath("./../model/piano-model/rwkv-final.pth"), te), \
        genrateTheSong(
            ctx, out_len, relpath("./../model/drumL20-D512-x0601/rwkv-final.pth"), te)


    music = GenMusic(pinio, drum)

    music.mix_lines(song_len).export("./output.mp3", format="mp3")
    print('output saved')
    return relpath('./output.mp3')


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", "--ctx", help="Enter the Context")
    arg_parser.add_argument(
        "-m", "--midi", help="Enter the Midi file path", type=str)
    arg_parser.add_argument(
        "-t", "--out_len", help="Enter the Length of unique output", default=300, type=int
    )
    arg_parser.add_argument(
        "-st", "--song_len", help="Enter the Length of song len", default=60, type=int
    )
    arg_parser.add_argument(
        "-te", "--temperature", help="Enter the temperature of model", default=1.0, type=float
    )
    args = arg_parser.parse_args()

    context = ''

    if args.midi is not None:
        with open(args.midi, 'rb') as f:
            context = convert_midi_bytes_to_str(None, ('', f.read()))[1][0]
    elif args.ctx is not None:
        context = args.ctx
    else:
        context = f"26:2 t8 26:0 26:2 t5 26:0 t30 26:9 t13 26:0 t7"
    
    print(context)
    mixTheSongs(args.song_len, context, args.out_len, args.temperature)
