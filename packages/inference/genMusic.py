import random
import os
import math
from midi2audio import FluidSynth
from pydub import AudioSegment
from src.tokenizer.midi_util import VocabConfig, convert_str_to_midi
import sys
sys.path.append("./../model")


SECOND = 1000


def relpath(p): return os.path.normpath(
    os.path.join(os.path.dirname(__file__), p))


class GenMusic:
    def __init__(self, data, dataDrum):
      
        tempo = random.randint(14, 18) * 5

        self.midi = convert_str_to_midi(" ".join(map(str, data)), tempo).save(
            relpath("./soundFont/mdiOut.mid")
        )
        fs = FluidSynth(relpath("./soundFont/SGM-v2.01-NicePianosGuitarsBass-V1.2.sf2"))
        fs.midi_to_audio(relpath("./soundFont/mdiOut.mid"),
                         relpath("./soundFont/output.flac"))
        self.pianoRoll=AudioSegment.from_file(relpath("./soundFont/output.flac"))

        fillName = (
            random.choice(os.listdir(relpath("./loops/vinyl")))
        )
        self.fill = AudioSegment.from_file(relpath(f"./loops/vinyl/{fillName}"))

        # handel drum
        convert_str_to_midi(" ".join(map(str, data)), 400, 9).save(
            relpath("./soundFont/mdiOutDrum.mid")
        )
        fs.midi_to_audio(relpath("./soundFont/mdiOutDrum.mid"),
                         relpath("./soundFont/output.flac"))
        self.drum = AudioSegment.from_file(relpath("./soundFont/output.flac"))

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
