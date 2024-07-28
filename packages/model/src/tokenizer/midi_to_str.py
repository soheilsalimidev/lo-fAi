import argparse
import io

import os
from typing import Iterable, List, Optional, Tuple, Union

import mido
from tqdm import tqdm

import midi_util
from midi_util import AugmentConfig, VocabConfig, FilterConfig


def convert_midi_bytes_to_str(cfg: VocabConfig, filter_cfg: FilterConfig, aug_cfg: AugmentConfig, data: Tuple[str, bytes]) -> Tuple[str, Union[str, List, None]]:
    print("dsfjds")
    filename, filedata = data
    
    try:
        mid = mido.MidiFile(file=io.BytesIO(filedata))
    except:
        print("dsfjds")
        return filename, filedata
    if mid.type not in (0, 1, 2):
        print("dsfjds")
        return filename, None
    if len(mid.tracks) == 0:
        
        return filename, filedata

    # if aug_cfg is not None:
    #     return filename, [*(midi_util.convert_midi_to_str(cfg, filter_cfg, mid, augment) for augment in aug_cfg.get_augment_values(filename))]
    print(mid)
    return filename, midi_util.convert_midi_to_str(cfg, filter_cfg, mid)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to a folder or archive containing MIDI files",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSONL file",
    )
    p.add_argument(
        "--vocab_config",
        type=str,
        default="./vocab_config.json",
        help="Path to vocab config file",
    )
    p.add_argument(
        "--filter_config",
        type=str,
        default="./filter_config.json",
        help="Path to filter config file",
    )
    p.add_argument(
        "--augment_config",
        type=str,
        help="Path to augment config file",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers to use for parallel processing",
    )
    args = p.parse_args()

    cfg = VocabConfig.from_json(args.vocab_config)

    filter_config = FilterConfig.from_json(args.filter_config)

    augment_config = None
    if args.augment_config is not None:
        augment_config = AugmentConfig.from_json(args.augment_config, cfg)
   
    with open(args.path, "rb") as f: 
     filebytes = f.read()
     print(convert_midi_bytes_to_str(
         cfg, filter_config, None, (os.path.basename(args.path), filebytes)))
