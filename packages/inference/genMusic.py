import sys
sys.path.append("./../model")
from src.tokenizer.midi_util import VocabConfig, convert_str_to_midi
from pydub import AudioSegment
from midi2audio import FluidSynth
import math
import os
import random


SECOND = 1000


class GenMusic:
    def __init__(self, data, dataDrum):
        cfg = VocabConfig.from_json(
            "./../model/src/tokenizer/vocab_config.json")
        tempo = random.randint(14, 18) * 5

        self.midi = convert_str_to_midi(cfg, " ".join(map(str, data)), tempo).save(
            "./soundFont/mdiOut.mid"
        )
        fs = FluidSynth("./soundFont/SGM-v2.01-NicePianosGuitarsBass-V1.2.sf2")
        fs.midi_to_audio("./soundFont/mdiOut.mid", "./soundFont/output.flac")
        self.pianoRoll = AudioSegment.from_file("./soundFont/output.flac")

        fillName = (
            random.choice(os.listdir("./loops/vinyl"))
        )
        self.fill = AudioSegment.from_file(f"./loops/vinyl/{fillName}")

        # handel drum
        convert_str_to_midi(cfg, " ".join(map(str, data)), 400, 9).save(
            "./soundFont/mdiOutDrum.mid"
        )
        fs.midi_to_audio("./soundFont/mdiOutDrum.mid",
                         "./soundFont/output.flac")
        self.drum = AudioSegment.from_file("./soundFont/output.flac")

        # self.drum = AudioSegment.from_file(f"./loops/drumloop{tempo}/{drum}")

    def mix_lines(self, music_len=60):
        self.pianoRoll = self.pianoRoll + 10

        self.fill = self.fill * \
            math.ceil(self.pianoRoll.duration_seconds /
                      self.fill.duration_seconds)
        self.drum = self.drum * \
            math.ceil(self.pianoRoll.duration_seconds /
                      self.drum.duration_seconds)
        music = self.pianoRoll.overlay(
            self.fill - 10,
        ).overlay(
            self.drum + 3,
        )
        music = music * math.ceil(music_len / music.duration_seconds)

        music = music.fade_out(2 * SECOND)

        return music
