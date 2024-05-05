import random

from midi_to_jsonl import midi_to_jsonl
from midi_util import FilterConfig, VocabConfig



######################
# midi_to_jsonl
######################
cfg = VocabConfig.from_json("./vocab_config.json")
filter_config = FilterConfig.from_json("./filter_config.json")
midi_to_jsonl( cfg,filter_config ,"./../midiData/piano/irishman-midi.zip" , "./../midiData/piano/piano.jsonl" , None , 8 )

###############
#shuffle data for 2 epoch
###################
N_EPOCH = 2

with open("./../midiData/piano.jsonl", "r", encoding="utf-8") as file:
    non_empty_lines = [line.strip() for line in file if line.strip()]

print(f"### Found {len(non_empty_lines)} non-empty lines in ./../midiData/piano.jsonl")

file = open("./../midiData/piano_temp.jsonl", "w", encoding="utf-8")
for i in range(N_EPOCH):
    print(f"Shuffle: {i+1} out of {N_EPOCH}")
    random.shuffle(non_empty_lines)
    for entry in non_empty_lines:
        file.write(entry + "\n")
file.close()

