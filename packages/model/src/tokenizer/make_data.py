import os
import sys

sys.path.insert(1, os.getcwd())

import argparse
from pathlib import Path

from json2binidx_tool import preprocess_data
from midi_to_jsonl import midi_to_jsonl
from midi_util import FilterConfig, VocabConfig


######################
# midi_to_jsonl
######################
def makeData(path: str, out: str):
    cfg = VocabConfig.from_json(Path("./vocab_config.json").absolute().__str__())
    filter_config = FilterConfig.from_json(Path("./filter_config.json").absolute().__str__())
    outPath = out + "out.jsonl"
    midi_to_jsonl(
        cfg,
        filter_config,
        path,
        outPath,
        None,
        8,
    )

    with open(path, "r", encoding="utf-8") as file:
        non_empty_lines = [line.strip() for line in file if line.strip()]
    
    print(f"### Found {len(non_empty_lines)} non-empty lines in {outPath}")
    preprocess_data.startTheProsses(
        outPath, path, Path("./tokenizer-midi/tokenizer.json").absolute().__str__()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--input-f",
        type=str,
        help="input path",
    )
    group.add_argument(
        "--out",
        type=str,
        help="out path",
    )
    args = parser.parse_args()
    makeData(args.input_f , args.out)

