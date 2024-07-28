import math
import os
import random

from midi2audio import FluidSynth
from pydub import AudioSegment
from src.tokenizer.midi_util import VocabConfig, convert_str_to_midi

soundfont_path = "./soundFont/florestan-piano.sf2"


presets = {1: {}, 2: {}, 3: {}}


SECOND = 1000


class GenMusic:
    def __init__(self, data):
        cfg = VocabConfig.from_json("./../model/src/tokenizer/vocab_config.json")
        tempo = random.randint(14, 18) * 5
        self.midi = convert_str_to_midi(cfg, " ".join(map(str, data)), tempo).save(
            "./soundFont/mdiOut.mid"
        )
        fs = FluidSynth("./soundFont/SGM-v2.01-NicePianosGuitarsBass-V1.2.sf2")
        fs.midi_to_audio("./soundFont/mdiOut.mid", "./soundFont/output.flac")
        self.pianoRoll = AudioSegment.from_file("./soundFont/output.flac")
        print(self.pianoRoll.duration_seconds)
        fillName = (
            random.choice(os.listdir("./loops/vinyl"))
            if random.randint(0, 10) > 5
            else random.choice(os.listdir("./loops/vinyl"))
        )
        self.fill = AudioSegment.from_file(f"./loops/vinyl/{fillName}")
        drum = random.choice(os.listdir(f"./loops/drumloop{tempo}"))
        self.drum = AudioSegment.from_file(f"./loops/drumloop{tempo}/{drum}")

    def mix_lines(self, music_len=60):
        self.pianoRoll = self.pianoRoll + 10
        music = self.pianoRoll.overlay(
            self.fill,
        ).overlay(
            self.drum - 15,
        )
        music = music * math.ceil(music_len / music.duration_seconds)

        music = music.fade_out(2 * SECOND)

        return music
