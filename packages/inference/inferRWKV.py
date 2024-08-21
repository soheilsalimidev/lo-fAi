from rwkv.utils import PIPELINE, PIPELINE_ARGS
from rwkv.model import RWKV
import argparse
import os
from genMusic import GenMusic


os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"


def genrateTheSong(ctx: str, lenOfOut: int, modelPath: str):
    model = RWKV(model=modelPath, strategy="cuda fp16")
    pipeline = PIPELINE(
        model, "./../model/src/tokenizer/tokenizer-midi/tokenizer.json")

    data = []

    args = PIPELINE_ARGS(
        temperature=1.5,
        top_p=0.9,
        top_k=100,  # top_k = 0 then ignore
        alpha_frequency=0.25,
        alpha_presence=0.25,
        alpha_decay=0.996,  # gradually decay the penalty
        token_ban=[],  # ban the generation of some tokens
        # stop generation whenever you see any token here
        token_stop=["<end>"],
        chunk_len=256,
    )  # split input into chunks to save VRAM (shorter -> slower)

    while len(data) <= lenOfOut:
        pipeline.generate(
            ctx, args=args, callback=lambda token: data.append(token))
        ctx = " ".join(map(str, data[:5]))
        print(len(data))

    return data


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

    pinio, drum= genrateTheSong(args.ctx, args.out_len,
                   "/home/arthur/AItuneCraft/packages/model/piano-model/rwkv-final.pth") ,  genrateTheSong(args.ctx, args.out_len,
                   "/home/arthur/AItuneCraft/packages/model/drumL20-D512-x0601/rwkv-final.pth")
    # pinio,  drum = ['t125', 't124', '29:0', 't1', '49:c', 't124', '49:0', 't1', '47:a', 't124', '47:0', 't1', '45:a', 't124', '45:0', 't1', '45:b', 't125', 't124', '45:0', 't1', '4e:a', 't124', '4e:0', 't1', '4c:c', 't125', 't125', 't124', '4c:0', 't1', '4a:b', 't125', 't124', '4a:0', 't1', '49:a', 't124', '49:0', 't1', '47:c', 't125', 't62', '47:0', 't1', '49:a', 't62', '49:0', 't1', '45:a', 't125', 't124', '45:0', 't1', '4a:a', 't62', '4a:0', 't1', '4c:a', 't62', '4c:0', 't1', '4e:c', 't125', 't62', '4e:0', 't1', '51:a', 't62', '51:0', 't1', '4f:a', 't124', '4f:0', 't1', '4e:b', 't125', 't124', '4e:0', 't1', '4c:a', 't124', '4c:0', 't1', '4a:c', 't125', 't62', '4a:0', 't1', '4c:a', 't62', '4c:0', 't1', '4a:a', 't124', '4a:0', 't1', '4a:b', 't62', '4a:0', 't125', 't62', '49:0', 't1', '49:a', 't62', '49:0', 't1', '49:a', 't125', 't124', '49:0', 't1', '47:c', 't125', 't62', '47:0', 't1', '47:a', 't62', '47:0', 't1', '49:a', 't125', 't62', '49:0', 't1', '47:a', 't62', '47:0', 't1', '45:c', 't124', '45:0', 't125', 't26', '42:a', 't122', '42:0', 't1', '44:a', 't125', 't62', '44:0', 't1', '45:a', 't62', '45:0', 't1', '47:c', 't124', '47:0', 't1', '47:a', 't124', '47:0', 't1', '45:a', 't125', 't125', 't124', '45:0', '', '', '4a:c', 't37', '45:a', '4a:0', 't37', '42:c', '45:0', 't75', '42:0', '45:a', 't37', '42:a', '45:0', 't37', '42:0', '43:c', 't37', '43:0', '47:a', 't37', '47:0', '4c:a', 't37', '4c:0', '4e:a', 't37', '4e:0', '4f:c', 't75', '4f:0', '53:a', 't56', '53:0',
    #                 '53:a', 't19', '51:c'], ['t1', '2c:8', 't10', '28:e', 't1', '33:0', 't1', '2c:0', 't2', '24:5', 't9', '28:0', 't4', '24:0', 't8', '28:8', '16:9', 't12', '28:0', '16:0', 't10', '24:8', 't1', '16:a', 't11', '24:0', 't1', '16:0', 't11', '26:6', '2a:6', 't13', '26:0', '2a:0', 't11', '2a:7', 't1', '24:7', 't11', '2a:0', 't1', '24:0', 't10', '25:b', 't1', '16:a', 't12', '25:0', 't1', '16:0', 't10', '2a:7', 't13', '2a:0', 't11', '24:8', 't2', '1a:9', 't10', '24:0', 't2', '1a:0', 't13', '2e:e', '28:f', 't12', '28:0', '36:0', 't10', '26:d', 't13', '26:0', 't8', '2c:9', 't3', '2a:9', 't10', '2c:0', 't3', '2a:0', 't9', '2a:6', 't2', 't2', '26:2', 't9', '2a:0', 't2', '26:0', 't8', '24:6', '2a:8', 't12', '24:0', '2a:0', 't11', '2b:e', '33:0', 't14', '2b:0', '28:0', 't2', '2c:0', 't10', '1a:b', 't13', '1a:0', 't4', '26:6', 't10', '26:0', '26:6', 't3', '26:0', 't5', '2c:9', 't2', '16:c', '24:8', 't10', '2c:0', 't2', '16:0', '24:0', 't24', '2a:8', 't13', '2a:0', 't4', '26:9', 't13', '26:0', 't7', '26:e', 't13', '26:0', 't6', '26:b', 't13', '26:0', 't4', '2c:a', 't3', '26:d', 't10', '2c:0', 't3', '26:0', 't6', '26:6', 't13', '26:0', 't6', '26:e', 't1', '2c:8', 't12', '26:0', 't1', '2c:0', 't5', '26:6', 't13', '26:0', 't8', '26:d', 't4', '2c:b', 't9', '26:0', 't3', '26:f', 't1', '2c:0', 't12', '26:0', 't6', '26:f', 't8', '2c:b', 't5', '26:0', 't7', '2b:b', 't1', '2c:0', 't12', '2b:0', 't3', '2b:e', 't13', '2b:0', 't8', '24:b', 't1', '37:f', 't11', '24:0', 't2', '37:0']

    music = GenMusic(pinio, drum)

    music.mix_lines(args.song_len).export("./output.mp3", format="mp3")
