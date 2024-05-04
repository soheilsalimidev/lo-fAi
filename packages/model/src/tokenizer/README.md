# run the following to get blindx
```sh
python3 ./tokenizer.py

python ./midi_to_jsonl.py --path ./../midiData/data/irishman-midi.zip --output ./../midiData/data/piano.jsonl --worker 4

python ./json2binidx_tool/tools/preprocess_data.py --input ./../midiData/piano/piano.jsonl --output-prefix ./../midiData --vocab ./tokenizer-midi/tokenizer.json --dataset-impl mmap --tokenizer-type HFTokenizer

```
