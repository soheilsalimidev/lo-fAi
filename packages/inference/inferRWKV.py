import sys
sys.path.append("./../model")
sys.path.append("./../model/src/tokenizer")

import random
from rwkv.utils import PIPELINE, PIPELINE_ARGS
from rwkv.model import RWKV
import argparse
import os
from genMusic import GenMusic
from src.tokenizer.midi_to_str import convert_midi_bytes_to_str



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

    # pinio,  drum = ['t125', 't124', '29:0', 't1', '49:c', 't124', '49:0', 't1', '47:a', 't124', '47:0', 't1', '45:a', 't124', '45:0', 't1', '45:b', 't125', 't124', '45:0', 't1', '4e:a', 't124', '4e:0', 't1', '4c:c', 't125', 't125', 't124', '4c:0', 't1', '4a:b', 't125', 't124', '4a:0', 't1', '49:a', 't124', '49:0', 't1', '47:c', 't125', 't62', '47:0', 't1', '49:a', 't62', '49:0', 't1', '45:a', 't125', 't124', '45:0', 't1', '4a:a', 't62', '4a:0', 't1', '4c:a', 't62', '4c:0', 't1', '4e:c', 't125', 't62', '4e:0', 't1', '51:a', 't62', '51:0', 't1', '4f:a', 't124', '4f:0', 't1', '4e:b', 't125', 't124', '4e:0', 't1', '4c:a', 't124', '4c:0', 't1', '4a:c', 't125', 't62', '4a:0', 't1', '4c:a', 't62', '4c:0', 't1', '4a:a', 't124', '4a:0', 't1', '4a:b', 't62', '4a:0', 't125', 't62', '49:0', 't1', '49:a', 't62', '49:0', 't1', '49:a', 't125', 't124', '49:0', 't1', '47:c', 't125', 't62', '47:0', 't1', '47:a', 't62', '47:0', 't1', '49:a', 't125', 't62', '49:0', 't1', '47:a', 't62', '47:0', 't1', '45:c', 't124', '45:0', 't125', 't26', '42:a', 't122', '42:0', 't1', '44:a', 't125', 't62', '44:0', 't1', '45:a', 't62', '45:0', 't1', '47:c', 't124', '47:0', 't1', '47:a', 't124', '47:0', 't1', '45:a', 't125', 't125', 't124', '45:0', '', '', '4a:c', 't37', '45:a', '4a:0', 't37', '42:c', '45:0', 't75', '42:0', '45:a', 't37', '42:a', '45:0', 't37', '42:0', '43:c', 't37', '43:0', '47:a', 't37', '47:0', '4c:a', 't37', '4c:0', '4e:a', 't37', '4e:0', '4f:c', 't75', '4f:0', '53:a', 't56', '53:0',
    #                 '53:a', 't19', '51:c'], ['<start>', '26:2', 't8', '26:0', '26:2', 't5', '26:0', 't30', '26:9', 't13', '26:0', 't7', '2c:a', 't2', '26:8', 't11', '2c:0', 't2', '26:0', 't12', '26:8', 't12', '26:0', 't9', '24:7', 't1', '2e:8', 't11', '24:0', 't1', '2e:0', 't11', '2a:5', 't13', '2a:0', 't9', '24:8', 't2', '2a:9', 't11', '24:0', 't2', '2a:0', 't11', '2a:4', 't13', '2a:0', 't10', '2a:7', 't1', '26:9', 't11', '2a:0', 't1', '26:0', 't8', '2a:7', 't3', '24:7', 't10', '2a:0', 't3', '24:0', 't8', '2a:8', 't13', '2a:0', 't11', '26:8', '2a:a', 't12', '26:0', '2a:0', 't9', '2a:8', 't13', '2a:0', 't9', '2a:7', '26:6', 't12', '2a:0', '26:0', 't6', '24:6', 't2', '2a:8', 't11', '24:0', 't2', '2a:0', 't11', '2a:6', 't4', '26:4', 't9', '2a:0', 't4', '26:0', 't7', '26:9', '2a:8', 't13', '26:0', '2a:0', 't10', '24:7', 't3', '2a:7', 't9', '24:0', 't3', '2a:0', 't9', '2a:9', 't13', '2a:0', 't11', '26:9', 't1', '2a:a', 't12', '26:0', 't1', '2a:0', 't12', '2a:6', 't13', '2a:0', 't11', '2a:7', 't13', '2a:0', 't11', '2a:8', 't1', '24:6', 't11', '2a:0', 't1', '24:0', 't11', '2a:3', 't12', '2a:0', 't11', '26:a', 't3', '2a:7', 't10', '26:0', 't3', '2a:0', 't11', '2a:5', 't2', '24:7', 't11', '2a:0', 't2', '24:0', 't6', '2a:8', 't13', '2a:0', 't9', '26:7', 't1', '2a:9', 't12', '26:0', 't1', '2a:0', 't11', '2a:6', 't13', '2a:0', 't8', '26:3', 't1', '2a:7', 't11', '26:0', 't1', '2a:0', 't8', '24:5', 't3', '2a:8', 't10', '24:0', 't3', '2a:0', 't8', '26:5', 't3', '2a:5', 't9', '26:0', 't3', '2a:0', 't8', '26:8', 't2', '2a:7', 't11', '24:0', 't14', '2b:d', 't12', '2b:0', '<end>']

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
        "-t", "--out_len", help="Enter the Length of unique output", default=100, type=int
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
        context = f"{random.randint(50,90):x}:{random.randint(1,16):x} t{random.randint(20,80)} " * 2

    mixTheSongs(args.song_len, context, args.out_len, args.temperature)
